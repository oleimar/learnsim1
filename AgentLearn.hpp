#ifndef AGLEARN_HPP
#define AGLEARN_HPP

#include "Agent.hpp"
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>

// The LearnSim program runs learning simulations
// Copyright (C) 2023  Olof Leimar
// See Readme.md for copyright notice

//**************************** class CompStim *****************************

// This struct represents an instance of a given type of a compound stimulus. An
// instance has a vector x of feature state values x[i] and a w_tr of
// contributions to the reward values from the different feature states, such
// that w_tr[i]*x[i] is the expected contribution from stimulus dimension i. The
// actual reward for an agent that selects the compound stimulus is log-normally
// distributed with this expectation.

template<typename flt>
struct CompStim {
public:
    using farr = std::vector<flt>;    
    CompStim(int a_nsd, flt a_sdR, int a_cstyp = 1) :
        nsd{a_nsd},
        sdR{a_sdR},
        muR{-sdR*sdR/2},
        cstyp{a_cstyp},
        w_tr(nsd, 0),
        x(nsd, 0) {}
    flt Reward(flt srn) const { 
        // srn should be a standard normal random number; we assume log-normal
        // variation in rewards
        return std::inner_product(w_tr.begin(), w_tr.end(), x.begin(),
            static_cast<flt>(0)) * std::exp(muR + sdR*srn);
    }
    int nsd;       // number of stimulus dimensions
    flt sdR;       // log-scale SD of the stochastic variation in reward
    flt muR;       // log-scale mean to compensate for stochastic variation
    int cstyp;     // indicates the type of compound stimulus
    farr w_tr;     // 'true' expected reward values for the stimulus dimensions
    farr x;        // state values (cues) for the stimulus dimensions
};


//************************** struct LearnStat *****************************

// This struct stores data on a learning bout, allowing for different kinds of
// learning agents

template<typename flt>
struct LearnStat {
public:
    using farr = std::vector<flt>;    
    int atyp;    // agent type
    int repl;    // replicate number
    int tstep;   // time step (round or trial) of learning
    int choice;  // choice (1 or 2)
    int cstyp1;  // type of compound stimulus 1
    int cstyp2;  // type of compound stimulus 2
    farr x1;     // feature values of compound stimulus 1
    farr x2;     // feature values of compound stimulus 2
    farr x;      // feature values of selected compound stimulus
    farr w_tr;   // true reward values for feature dimensions
    farr w;      // estimates of reward values for feature dimensions
    farr alph;   // learning rates for features
    flt Rew;     // perceived reward
    flt Q1;      // estimated value compound stimulus 1
    flt Q2;      // estimated value compound stimulus 2
    flt Q;       // estimated value selected compound stimulus
    flt Q1tr;    // true expected value compound stimulus 1
    flt Q2tr;    // true expected value compound stimulus 2
    flt delt;    // 'prediction error' (TD error)
};


//*************************************************************************

// The class below simulates learning using different meta-learning
// approaches. A learning agent experiences a number of bouts or trials. In
// each bout there is a choice between two compound stimuli, each
// characterised by feature states x[i] for one or more stimulus dimensions,
// with i = 1, ..., nsd. The feature states can either be quantitative, like
// the size of a compound stimulus, or can be the absence/presence of a
// particular feature, which is indicated as 0/1 for the corresponding x[i].

// The time sequence for a bout is, first, that the agent computes the estimated
// reward values of each compound stimulus (based of the estimated values of
// feature states), then the agent selects one of the compound stimuli, and
// perceives a reward. The resulting prediction error is used to first update
// the learning rates (if the type of agent has dynamically determined learning
// rates), and then to update the estimated reward values w[i] of the features
// present in the selected compound stimulus.

// In each round an agent is faced with selecting between two compound
// stimuli.


//***************************** class AgLearn ******************************

// A general learning class for an agent

template<typename flt, typename Agent>
class AgLearn {
public:
    using mlp_type = typename Agent::mlp_type;
    using farr = std::vector<flt>;    
    using stat_type = LearnStat<flt>;
    using vs_type = std::vector<stat_type>;
    using comp_stim = CompStim<flt>;
    using vcs_type = std::vector<comp_stim>;
    using rand_eng = std::mt19937;
    using rand_uni = std::uniform_real_distribution<flt>;
    using rand_int = std::uniform_int_distribution<int>;
    using rand_norm = std::normal_distribution<flt>;
    using rand_discr = std::discrete_distribution<int>;
    AgLearn(int a_nsd,
              int a_mn1,
              int a_mx1,
              int a_mn2,
              int a_mx2,
              int a_atyp,
              int a_repl,
              int a_T,
              flt a_sdR,
              flt a_sdx,
              flt a_w0,
              flt a_alph0,
              flt a_omega,
              const vcs_type& a_vcs,
              const mlp_type& a_mlp) :
        nsd{a_nsd},
        mn1{a_mn1},
        mx1{a_mx1},
        mn2{a_mn2},
        mx2{a_mx2},
        atyp{a_atyp},
        repl{a_repl},
        T{a_T},
        sdR{a_sdR},
        sdx{a_sdx},
        w0{a_w0},
        alph0{a_alph0},
        omega{a_omega},
        agent(nsd, w0, alph0),
        vcs{a_vcs},
        mlp{a_mlp} { 
            stat.reserve(2*T); 
        }
    const vs_type& Get_stat() const { return stat; }
    void Learn(rand_eng& eng);
    void Add_stat(int tstep, const comp_stim& cs1, const comp_stim& cs2);
    int nsd;         // number of stimulus dimensions
    int mn1;         // min of first set of compound stimuli
    int mx1;         // max of first set of compound stimuli
    int mn2;         // min of second set of compound stimuli
    int mx2;         // max of second set of compound stimuli
    int atyp;        // agent type
    int repl;        // replicate number (of agent type)
    int T;           // number of learning rounds
    flt sdR;         // sd of random reward variation
    flt sdx;         // sd of variation in x for first dimension
    flt avx1;        // mean of first dim of x for cs type 1
    flt avx2;        // mean of first dim of x for cs type 2
    flt w0;          // starting estimated feature value
    flt alph0;       // starting feature learning rate
    flt omega;       // parameter giving choice prob from values
    Agent agent;     // learning agent
    const vcs_type& vcs; // compound stimuli
    const mlp_type& mlp; // meta learning paramters
    vs_type stat;    // learning statistics
};

template<typename flt, typename Agent>
void AgLearn<flt, Agent>::Learn(rand_eng& eng)
{
    // Simulate learning by Rescorla-Wagner agent over 2*T rounds. 
    rand_uni uni(0, 1);
    rand_norm nrmx(-sdx*sdx/2, sdx); // random variation in first dimension
    rand_norm nrm(0, 1);    // standard normal random variation
    rand_int ri1(mn1, mx1); // uniform from mn1 to mx1
    rand_int ri2(mn2, mx2); // uniform from mn2 to mx2

    // run through 2*T rounds of learning 
    for (int tstep = 1; tstep <= 2*T; ++tstep) {
        // set the two compound stimuli for this round
        int typ1 = 0;
        int typ2 = 1;
        if (tstep <= T) {
            typ1 = ri1(eng);
            typ2 = ri1(eng);
        } else {
            typ1 = ri2(eng);
            typ2 = ri2(eng);
        }
        comp_stim cs1 = vcs[typ1 - 1];
        comp_stim cs2 = vcs[typ2 - 1];
        // // NOTE: temporary change
        // if (tstep <= T) {
        //     // introduce log-normal random variation in x[0]
        //     cs1.x[0] *= std::exp(nrmx(eng));
        //     cs2.x[0] *= std::exp(nrmx(eng));
        // } else {
        //     // change the value of dim 2
        //     cs1.w_tr[1] = -1.5;
        //     cs2.w_tr[1] = -1.5;
        // }
        // introduce log-normal random variation in x[0]
        cs1.x[0] *= std::exp(nrmx(eng));
        cs2.x[0] *= std::exp(nrmx(eng));
        // put the features states into the agent
        for (int i = 0; i < nsd; ++i) {
            agent.x1[i] = cs1.x[i];
            agent.x2[i] = cs2.x[i];
        }
        // compute estimated reward values
        agent.Q1 = agent.Qval(agent.x1);
        agent.Q2 = agent.Qval(agent.x2);
        // agent choice (based on estimated reward values)
        agent.choice = (uni(eng) < agent.PrCS1(omega)) ? 1 : 2;
        comp_stim& cs = (agent.choice == 1) ? cs1 : cs2;
        // get reward from selected compound stimulus
        agent.Rew = cs.Reward(nrm(eng));
        // store x and Q in agent
        agent.x = cs.x;
        agent.Q = (agent.choice == 1) ? agent.Q1 : agent.Q2;
        // prediction error
        agent.delt = agent.Rew - agent.Q;
        // update learning rates
        agent.Update_alph(mlp);
        // update estimated feature values
        agent.Update_w();
        // store learning statistics for this round
        Add_stat(tstep, cs1, cs2);
    }
}

// Add data from current round to the learning statistics container 
template<typename flt, typename Agent>
void AgLearn<flt, Agent>::Add_stat(int tstep, const comp_stim& cs1, const comp_stim& cs2) 
{
    stat_type st;
    st.atyp = atyp;
    st.repl = repl;
    st.tstep = tstep;
    st.choice = agent.choice;
    st.cstyp1 = cs1.cstyp;
    st.cstyp2 = cs2.cstyp;
    st.x1 = agent.x1;
    st.x2 = agent.x2;
    st.x = agent.x;
    st.w_tr = (st.choice == 1) ? cs1.w_tr : cs2.w_tr; 
    st.w = agent.w;
    st.alph = agent.alph;
    st.Rew = agent.Rew;
    st.Q1 = agent.Q1;
    st.Q2 = agent.Q2;
    st.Q = agent.Q;
    st.Q1tr = std::inner_product(cs1.w_tr.begin(), cs1.w_tr.end(), 
                                   cs1.x.begin(), static_cast<flt>(0));
    st.Q2tr = std::inner_product(cs2.w_tr.begin(), cs2.w_tr.end(), 
                                   cs2.x.begin(), static_cast<flt>(0));
    st.delt = agent.delt;
    stat.push_back(st);
}


#endif // AGLEARN_HPP
