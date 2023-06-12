#ifndef LEARNSIM_HPP
#define LEARNSIM_HPP

// #ifdef _OPENMP
// #define PARA_RUN
// #endif

#include "Agent.hpp"
#include "AgentLearn.hpp"
#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <random>
#include <algorithm>

// The LearnSim program runs learning simulations
// Copyright (C) 2023  Olof Leimar
// See Readme.md for copyright notice


using flt = double;

//*************************** Class InpData ****************************

// This class is used to 'package' input data in a single place; the
// constructor extracts data from an input file

class InpData {
public:
    using flt = double;         // the TOML class requires doubles
    std::size_t max_num_thrds;  // max number of threads to use
    int nsd;                    // number of stimulus dimensions
    int ncs;                    // number of compound stimuli
    int mn1;                    // min of first set of compound stimuli
    int mx1;                    // max of first set of compound stimuli
    int mn2;                    // min of second set of compound stimuli
    int mx2;                    // max of second set of compound stimuli
    int atyp;                   // learning agent type
    int nrepl;                  // number of replicate learning agents
    int T;                      // number of learning rounds
    flt sdR;                    // sd of random reward variation
    flt sdx;                    // sd of variation in x for first dimension
    flt w0;                     // starting estimated value of feature
    flt kapp0;                  // starting learning rate
    flt alph0;                  // starting value for alph learning element
    flt sigm0;                  // starting value for sigm learning element
    flt beta0;                  // starting value for beta learning element
    flt bet0;                   // starting value for bet
    flt h0;                     // starting value for h quantity
    flt omega;                  // parameter giving choice prob from values
    flt gam_a;                  // meta-learning rate for alph element
    flt gam_s;                  // meta-learning rate for sigm element
    flt mu;                     // meta-learning rate for AS and ID approach
    flt tau;                    // meta-learning par for AS approach
    std::string CSName;         // file name for CSV file for comp stims
    std::string h5OutName;      // file name for output of population

    std::string InpName;  // Name of input data file
    bool OK;              // Whether input data has been successfully read

    InpData(const char* filename);
};


//***************************** Class LearnSim ******************************

class LearnSim {
public:
    using farr = std::vector<flt>;
    using stat_type = LearnStat<flt>;
    using vs_type = std::vector<stat_type>;
    using comp_stim = CompStim<flt>;
    using vcs_type = std::vector<comp_stim>;
    using RWagl_type = AgLearn<flt, RWagent<flt, RWmlp<flt>>>;
    using RWmlp_type = RWmlp<flt>;
    using PMagl_type = AgLearn<flt, PMagent<flt, PMmlp<flt>>>;
    using PMmlp_type = PMmlp<flt>;
    using ASagl_type = AgLearn<flt, ASagent<flt, ASmlp<flt>>>;
    using ASmlp_type = ASmlp<flt>;
    using IDagl_type = AgLearn<flt, IDagent<flt, IDmlp<flt>>>;
    using IDmlp_type = IDmlp<flt>;
    using f_type = std::vector<flt>;
    using i_type = std::vector<int>;
    using ui_type = std::vector<unsigned>;
    using rand_eng = std::mt19937;
    using vre_type = std::vector<rand_eng>;
    // using rand_int = std::uniform_int_distribution<int>;
    using rand_uni = std::uniform_real_distribution<flt>;
    using rand_norm = std::normal_distribution<flt>;
    using rand_discr = std::discrete_distribution<int>;
    LearnSim(const InpData& lsid);
    void Run();
    void h5_write_stat(const std::string& outfilename) const;
private:
    InpData id;
    int nsd;                    // number of stimulus dimensions
    int ncs;                    // number of compund stimulus types to use
    int mn1;                    // min of first set of compound stimuli
    int mx1;                    // max of first set of compound stimuli
    int mn2;                    // min of second set of compound stimuli
    int mx2;                    // max of second set of compound stimuli
    int atyp;                   // learning agent type
    int nrepl;                  // number of learning replicates per agent
    int T;                      // number of learning rounds per replicate
    flt sdR;                    // sd of random reward variation
    flt sdx;                    // sd of variation in x for first dimension
    flt w0;                     // starting estimated value of feature
    flt alph0;                  // starting learning rate
    flt omega;                  // parameter giving choice prob from values
    flt gam_a;                  // meta-learning rate for aPM element
    flt gam_s;                  // meta-learning rate for sPM element
    flt mu;                     // meta-learning rate for MA and ID approach
    flt tau;                    // meta-learning par for AS approach
    std::size_t num_thrds;
    vcs_type vcs;               // vector of compound stimuli to use
    ui_type sds;                // seeds for random numbers
    vre_type vre;               // random number engines
    vs_type stat;               // learning statistics
};

#endif // LEARNSIM_HPP
