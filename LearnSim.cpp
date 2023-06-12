#include <limits>
#include "cpptoml.h"   // read input parameters from TOML file
#include "rapidcsv.h"  // read compound stimuli data from CSV file
#include "LearnSim.hpp"
#include "hdf5code.hpp"
#include "Utils.hpp"
#include <algorithm>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>

// The LearnSim program runs learning simulations
// Copyright (C) 2023  Olof Leimar
// See Readme.md for copyright notice

// #ifdef PARA_RUN
// #include <omp.h>
// #endif

//************************** Read and ReadArr ****************************

// convenience functions to read from TOML input file

// this template function can be used for any type of single value
template<typename T>
void Get(std::shared_ptr<cpptoml::table> infile,
         T& value, const std::string& name)
{
    auto val = infile->get_as<T>(name);
    if (val) {
        value = *val;
    } else {
        std::cerr << "Read failed for identifier " << name << "\n";
    }
}

// this template function can be used for a vector or array (but there is no
// checking how many elements are read)
template<typename It>
void GetArr(std::shared_ptr<cpptoml::table> infile,
            It beg, const std::string& name)
{
    using valtype = typename std::iterator_traits<It>::value_type;
    auto vp = infile->get_array_of<valtype>(name);
    if (vp) {
        std::copy(vp->begin(), vp->end(), beg);
    } else {
        std::cerr << "Read failed for identifier " << name << "\n";
    }
}


//**************************** class InpData *****************************

InpData::InpData(const char* filename) :
      OK(false)
{
    auto idat = cpptoml::parse_file(filename);
    Get(idat, max_num_thrds, "max_num_thrds");
    Get(idat, nsd, "nsd");
    Get(idat, ncs, "ncs");
    Get(idat, mn1, "mn1");
    Get(idat, mx1, "mx1");
    Get(idat, mn2, "mn2");
    Get(idat, mx2, "mx2");
    Get(idat, atyp, "atyp");
    Get(idat, nrepl, "nrepl");
    Get(idat, T, "T");
    Get(idat, sdR, "sdR");
    Get(idat, sdx, "sdx");
    Get(idat, w0, "w0");
    Get(idat, alph0, "alph0");
    Get(idat, omega, "omega");
    Get(idat, gam_a, "gam_a");
    Get(idat, gam_s, "gam_s");
    Get(idat, mu, "mu");
    Get(idat, tau, "tau");
    Get(idat, CSName, "CSName");
    Get(idat, h5OutName, "h5OutName");
    InpName = std::string(filename);
    OK = true;
}


//****************************** Class LearnSim *****************************

LearnSim::LearnSim(const InpData& lsid) :
    id{lsid},
    nsd{id.nsd},
    ncs{id.ncs},
    mn1{id.mn1},
    mx1{id.mx1},
    mn2{id.mn2},
    mx2{id.mx2},
    atyp{id.atyp},
    nrepl{id.nrepl},
    T{id.T},
    sdR{static_cast<flt>(id.sdR)},
    sdx{static_cast<flt>(id.sdx)},
    w0{static_cast<flt>(id.w0)},
    alph0{static_cast<flt>(id.alph0)},
    omega{static_cast<flt>(id.omega)},
    gam_a{static_cast<flt>(id.gam_a)},
    gam_s{static_cast<flt>(id.gam_s)},
    mu{static_cast<flt>(id.mu)},
    tau{static_cast<flt>(id.tau)},
    num_thrds{1}
{
    // decide on number of threads for parallel processing
#ifdef PARA_RUN
    num_thrds = omp_get_max_threads();
    if (num_thrds > id.max_num_thrds) num_thrds = id.max_num_thrds;
    std::cout << "Number of threads: "
              << num_thrds << '\n';
    int threadn = omp_get_thread_num();
#else
    int threadn = 0;
#endif
    // generate one seed for each thread
    sds.resize(num_thrds);
    std::random_device rd;
    for (unsigned i = 0; i < num_thrds; ++i) {
        sds[i] = rd();
        vre.push_back(rand_eng(sds[i]));
    }

    // construct global container of compound stimuli
    comp_stim cs0(nsd, sdR);
    vcs.resize(ncs, cs0);
    // read data for compound stimuli from CSV file
    rapidcsv::Document CSdat(id.CSName);
    for (int k = 0; k < ncs; ++k) {
        comp_stim& cs = vcs[k];
        cs.nsd = nsd;
        cs.sdR = sdR;
        cs.cstyp = k + 1;
        // first column of CSV file contains w_tr values
        cs.w_tr = CSdat.GetColumn<flt>(0);
        // features for successive compound stimuli in remaining columns
        cs.x = CSdat.GetColumn<flt>(k + 1);
    }

    // Note concerning thread safety: in order to avoid possible problems with
    // multiple threads, the std::vector container stat is allocated once and
    // for all here, and thread-local data are then copied into position in it
    // (thus avoiding potentially unsafe push_back and insert).

    // make room for nagt*nrepl*T LearnStat records in stat
    // stat.resize(nagt*nrepl*T);
    // alternatively
    stat.reserve(nrepl*2*T);
}

void LearnSim::Run()
{
    Timer timer(std::cout);
    timer.Start();
    ProgressBar PrBar(std::cout, nrepl);

    // use parallel for processing over the learning replicates
#pragma omp parallel for num_threads(num_thrds)
    for (int repl = 1; repl <= nrepl; ++repl) {
#ifdef PARA_RUN
        int threadn = omp_get_thread_num();
#else
        int threadn = 0;
#endif
        // thread-local random number engine
        rand_eng& eng = vre[threadn];
        // perform learning for current agent type and repl
        if (atyp == 1) {
            RWmlp_type mlp;
            RWagl_type agl(nsd, mn1, mx1, mn2, mx2, 
                atyp, repl, T, sdR, sdx, 
                w0, alph0, omega, vcs, mlp);
            agl.Learn(eng);
            const vs_type& st = agl.stat;
            stat.insert(stat.end(), st.begin(), st.end());
        } else if (atyp == 2) {
            ASmlp_type mlp;
            mlp.mu = mu;
            mlp.tau = tau;
            ASagl_type agl(nsd, mn1, mx1, mn2, mx2, 
                atyp, repl, T, sdR, sdx, 
                w0, alph0, omega, vcs, mlp);
            agl.Learn(eng);
            const vs_type& st = agl.stat;
            stat.insert(stat.end(), st.begin(), st.end());
        } else if (atyp == 3) {
            IDmlp_type mlp;
            mlp.mu = mu;
            IDagl_type agl(nsd, mn1, mx1, mn2, mx2, 
                atyp, repl, T, sdR, sdx, 
                w0, alph0, omega, vcs, mlp);
            agl.Learn(eng);
            const vs_type& st = agl.stat;
            stat.insert(stat.end(), st.begin(), st.end());
        } else if (atyp == 4) {
            PMmlp_type mlp;
            mlp.gam_a = gam_a;
            mlp.gam_s = gam_s;
            PMagl_type agl(nsd, mn1, mx1, mn2, mx2, 
                atyp, repl, T, sdR, sdx, 
                w0, alph0, omega, vcs, mlp);
            agl.Learn(eng);
            const vs_type& st = agl.stat;
            stat.insert(stat.end(), st.begin(), st.end());
        }
        ++PrBar;
    } // end of parallel for (over replicates)
    PrBar.Final();
    timer.Stop();
    timer.Display();
    h5_write_stat(id.h5OutName);
}

void LearnSim::h5_write_stat(const std::string& outfilename) const
{
    h5W h5(outfilename);
    unsigned slen = stat.size();
    // std::vector to hold int member
    i_type ival(slen);
    // atyp
    std::transform(stat.begin(), stat.end(), ival.begin(),
                   [](const stat_type& st) -> int
                   { return st.atyp; });
    h5.write_int("atyp", ival);
    // repl
    std::transform(stat.begin(), stat.end(), ival.begin(),
                   [](const stat_type& st) -> int
                   { return st.repl; });
    h5.write_int("repl", ival);
    // tstep
    std::transform(stat.begin(), stat.end(), ival.begin(),
                   [](const stat_type& st) -> int
                   { return st.tstep; });
    h5.write_int("tstep", ival);
    // choice
    std::transform(stat.begin(), stat.end(), ival.begin(),
                   [](const stat_type& st) -> int
                   { return st.choice; });
    h5.write_int("choice", ival);
    // cstyp1
    std::transform(stat.begin(), stat.end(), ival.begin(),
                   [](const stat_type& st) -> int
                   { return st.cstyp1; });
    h5.write_int("cstyp1", ival);
    // cstyp2
    std::transform(stat.begin(), stat.end(), ival.begin(),
                   [](const stat_type& st) -> int
                   { return st.cstyp2; });
    h5.write_int("cstyp2", ival);
    // // container to hold integer arrays
    // std::vector<i_type> ivals(slen, i_type(nsd));
    // container to hold flt arrays
    std::vector<f_type> fvals(slen, f_type(nsd));
    // write feature values of compound stimulus 1
    for (int l = 0; l < slen; ++l) {
        const farr& x1 = stat[l].x1;
        for (int i = 0; i < nsd; ++i) {
            fvals[l][i] = x1[i];
        }
    }
    h5.write_flt_arr("x1", fvals);
    // write feature values of compound stimulus 2
    for (int l = 0; l < slen; ++l) {
        const farr& x2 = stat[l].x2;
        for (int i = 0; i < nsd; ++i) {
            fvals[l][i] = x2[i];
        }
    }
    h5.write_flt_arr("x2", fvals);
    // write feature values of selected compound stimulus
    for (int l = 0; l < slen; ++l) {
        const farr& x = stat[l].x;
        for (int i = 0; i < nsd; ++i) {
            fvals[l][i] = x[i];
        }
    }
    h5.write_flt_arr("x", fvals);
    // write true values of features of chosen stimulus
    for (int l = 0; l < slen; ++l) {
        const farr& w_tr = stat[l].w_tr;
        for (int i = 0; i < nsd; ++i) {
            fvals[l][i] = w_tr[i];
        }
    }
    h5.write_flt_arr("w_tr", fvals);
    // write estimated values of features
    for (int l = 0; l < slen; ++l) {
        const farr& w = stat[l].w;
        for (int i = 0; i < nsd; ++i) {
            fvals[l][i] = w[i];
        }
    }
    h5.write_flt_arr("w", fvals);
    // write learning rates
    for (int l = 0; l < slen; ++l) {
        const farr& alph = stat[l].alph;
        for (int i = 0; i < nsd; ++i) {
            fvals[l][i] = alph[i];
        }
    }
    h5.write_flt_arr("alph", fvals);
    // std::vector to hold flt member
    f_type fval(slen);
    // Rew
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.Rew; });
    h5.write_flt("Rew", fval);
    // Q1
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.Q1; });
    h5.write_flt("Q1", fval);
    // Q2
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.Q2; });
    h5.write_flt("Q2", fval);
    // Q
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.Q; });
    h5.write_flt("Q", fval);
    // Q1tr
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.Q1tr; });
    h5.write_flt("Q1tr", fval);
    // Q2tr
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.Q2tr; });
    h5.write_flt("Q2tr", fval);
    // delt
    std::transform(stat.begin(), stat.end(), fval.begin(),
                   [](const stat_type& st) -> flt
                   { return st.delt; });
    h5.write_flt("delt", fval);
}
