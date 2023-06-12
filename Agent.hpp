#ifndef AGENT_HPP
#define AGENT_HPP

#include <string>
#include <vector>
#include <ostream>
#include <istream>
#include <cmath>
#include <numeric>
#include <algorithm>

// The LearnSim program runs learning simulations
// Copyright (C) 2023  Olof Leimar
// See Readme.md for copyright notice

// Concerning learning rates for the agents, the code below uses a notation
// where alph is a learning rate without the contribution from the cue x (this
// seems to be a 'standard convention' in the literature)


//************************* struct RWagent ******************************

// Agent with Rescorla-Wagner learning

// struct with meta-learning parameters
template<typename flt>
struct RWmlp {
    flt nopar; // there are no meta-learning parameters for RW
};

template<typename flt, typename mlp>
struct RWagent {
public:
    using mlp_type = mlp;
    using farr = std::vector<flt>;    
    RWagent(int a_nsd = 4,
            flt w0 = 0,
            flt alph0 = 1) :
        nsd{a_nsd},
        w(nsd, w0),
        alph(nsd, alph0),
        x1(nsd, 0),
        x2(nsd, 0),
        x(nsd, 0) {}
    // return the estimated reward value of feature vector x
    flt Qval(const farr& xv) const 
    { 
        return std::inner_product(w.begin(), w.end(), xv.begin(), 
        static_cast<flt>(0));
    }
    // return the probability of selecting compound stimulus 1 (soft-max)
    flt PrCS1(flt omega) const { return 1/(1 + std::exp(-omega*(Q1 - Q2))); }
    // Learning update of w, using current learning rates
    void Update_w();
    void Update_alph(const mlp& lp);
    // public data members
    int nsd;     // number of stimulus dimensions
    farr w;      // estimates of reward values for features
    farr alph;   // learning rate
    farr x1;     // features of compound stimulus 1
    farr x2;     // features of compound stimulus 2
    farr x;      // features of selected compound stimulus
    flt Rew;     // perceived reward
    flt Q1;      // estimated value compound stimulus 1
    flt Q2;      // estimated value compound stimulus 2
    flt Q;       // estimated value of selected compound stimulus
    flt delt;    // prediction error (TD error)
    int choice;  // choice of compound stimulus (1 or 2)
};

template<typename flt, typename mlp>
void RWagent<flt, mlp>::Update_w()
{
    for (int i = 0; i < nsd; ++i) {
        w[i] += alph[i]*x[i]*delt;
    }
}

template<typename flt, typename mlp>
void RWagent<flt, mlp>::Update_alph(const mlp& lp)
{
    // there is no update of learning rates for Rescorla-Wagner
}


//************************* struct ASagent ******************************

// Agent with Autostep (Mahmood et al. 2012) learning

// struct with meta-learning parameters
template<typename flt>
struct ASmlp {
    flt mu;
    flt tau;
};

template<typename flt, typename mlp>
struct ASagent {
public:
    using mlp_type = mlp;
    using farr = std::vector<flt>;
    // constructor
    ASagent(int a_nsd = 4,
            flt w0 = 0,
            flt alph0 = 1) :
        nsd{a_nsd},
        w(nsd, w0),
        alph(nsd, alph0),
        v(nsd, 0),
        h(nsd, 0),
        x1(nsd, 0),
        x2(nsd, 0),
        x(nsd, 0) {}
    // return the estimated reward value of feature vector x
    flt Qval(const farr& xv) const 
    { 
        return std::inner_product(w.begin(), w.end(), xv.begin(), 
        static_cast<flt>(0));
    }
    // return the probability of selecting compound stimulus 1 (soft-max)
    flt PrCS1(flt omega) const { return 1/(1 + std::exp(-omega*(Q1 - Q2))); }
    // Learning update of w, using current learning rates
    void Update_w();
    // Autostep learning update
    void Update_alph(const mlp& lp);
    // public data members
    int nsd;     // number of stimulus dimensions
    farr w;      // estimates of reward values for features
    farr alph;   // learning rate
    farr v;      // quantity introduced by Mahmood et al. 2012
    farr h;      // quantity introduced by Mahmood et al. 2012
    farr x1;     // feature states of compound stimulus 1
    farr x2;     // feature states of compound stimulus 2
    farr x;      // feature states of selected compound stimulus
    flt Rew;     // perceived reward
    flt Q1;      // estimated value compound stimulus 1
    flt Q2;      // estimated value compound stimulus 2
    flt Q;       // estimated value of selected compound stimulus
    flt delt;    // prediction error (TD error)
    int choice;  // choice of compound stimulus (1 or 2)
};

template<typename flt, typename mlp>
void ASagent<flt, mlp>::Update_w()
{
    for (int i = 0; i < nsd; ++i) {
        w[i] += alph[i]*x[i]*delt;
    }
}

template<typename flt, typename mlp>
void ASagent<flt, mlp>::Update_alph(const mlp& lp)
{
    // this is the Autostep algorithm of Table 1 in Mahmood et al 2012
    flt mu = lp.mu;
    flt tau = lp.tau;
    flt sum = 0;
    for (int i = 0; i < nsd; ++i) {
        flt dxh = std::abs(delt*x[i]*h[i]);
        v[i] = std::max(dxh, v[i] + (dxh - v[i])*alph[i]*x[i]*x[i]/tau);
        if (v[i] != 0) {
            alph[i] *= std::exp(mu*delt*h[i]*x[i]/v[i]);
        }
        sum += alph[i]*x[i]*x[i];
    }
    flt M = std::max(sum, static_cast<flt>(1));
    for (int i = 0; i < nsd; ++i) {
        alph[i] /= M;
        h[i] = h[i]*(1 - alph[i]*x[i]*x[i]) + alph[i]*x[i]*delt;
    }
}


//************************* struct IDagent ******************************

// Agent with IDBD (Sutton 1992a) learning

// struct with meta-learning parameters
template<typename flt>
struct IDmlp {
    flt mu;
};

template<typename flt, typename mlp>
struct IDagent {
public:
    using mlp_type = mlp;
    using farr = std::vector<flt>;
    // constructor
    IDagent(int a_nsd = 4,
            flt w0 = 0,
            flt alph0 = 1) :
        nsd{a_nsd},
        w(nsd, w0),
        alph(nsd, alph0),
        h(nsd, 0),
        x1(nsd, 0),
        x2(nsd, 0),
        x(nsd, 0) {}
    // return the estimated reward value of feature vector x
    flt Qval(const farr& xv) const 
    { 
        return std::inner_product(w.begin(), w.end(), xv.begin(), 
        static_cast<flt>(0));
    }
    // return the probability of selecting compound stimulus 1 (soft-max)
    flt PrCS1(flt omega) const { return 1/(1 + std::exp(-omega*(Q1 - Q2))); }
    // Learning update of w, using current learning rates
    void Update_w();
    // ReLU function
    flt ReLU(flt X) const { return (X > 0) ? X : 0; }
    // IDBD learning update
    void Update_alph(const mlp& lp);
    // public data members
    int nsd;     // number of stimulus dimensions
    farr w;      // estimates of reward values for features
    farr alph;   // learning rate
    farr h;      // quantity introduced by Sutton 1992a
    farr x1;     // feature states of compound stimulus 1
    farr x2;     // feature states of compound stimulus 2
    farr x;      // feature states of selected compound stimulus
    flt Rew;     // perceived reward
    flt Q1;      // estimated value compound stimulus 1
    flt Q2;      // estimated value compound stimulus 2
    flt Q;       // estimated value of selected compound stimulus
    flt delt;    // prediction error (TD error)
    int choice;  // choice of compound stimulus (1 or 2)
};

template<typename flt, typename mlp>
void IDagent<flt, mlp>::Update_w()
{
    for (int i = 0; i < nsd; ++i) {
        w[i] += alph[i]*x[i]*delt;
    }
}

template<typename flt, typename mlp>
void IDagent<flt, mlp>::Update_alph(const mlp& lp)
{
    // This is the IDBD algorithm of Sutton 1992a, also described in Sutton
    // 1992b, and in Mahmood et al 2012; note that the starting value of alph
    // needs to be right to prevent divergence of the algorithm
    flt mu = lp.mu;
    for (int i = 0; i < nsd; ++i) {
        alph[i] *= std::exp(mu*delt*h[i]*x[i]);
        h[i] = h[i]*ReLU(1 - alph[i]*x[i]*x[i]) + alph[i]*x[i]*delt;
        // see eq. (3) in Mahmood et al. 2012, eqs. (17, 20) in Sutton 1992b, or
        // eqs. (4, 5) and fig. 2 in Sutton 1992a
    }
}


//************************* struct PMagent ******************************

// Agent with Pearce-Macintosh learning

// struct with meta-learning parameters
template<typename flt>
struct PMmlp {
    flt gam_a;
    flt gam_s;
};

template<typename flt, typename mlp>
struct PMagent {
public:
    using mlp_type = mlp;
    using farr = std::vector<flt>;    
    // constructor
    PMagent(int a_nsd = 4,
            flt w0 = 0,
            flt alph0 = 1) :
        nsd{a_nsd},
        w(nsd, w0),
        alph(nsd, alph0),
        aPM(nsd, alph0),
        sPM(nsd, 1),
        x1(nsd, 0),
        x2(nsd, 0),
        x(nsd, 0) {}
    // return the estimated reward value of feature vector x
    flt Qval(const farr& xv) const 
    { 
        return std::inner_product(w.begin(), w.end(), xv.begin(), 
        static_cast<flt>(0));
    }
    // return the probability of selecting compound stimulus 1 (soft-max)
    flt PrCS1(flt omega) const { return 1/(1 + std::exp(-omega*(Q1 - Q2))); }
    // Learning update of w, using current learning rates
    void Update_w();
    // Pearce-Mackintosh 'unified' meta learning update
    void Update_alph(const mlp& lp);
    // public data members
    int nsd;     // number of stimulus dimensions
    farr w;      // estimates of reward values for features
    farr alph;   // learning rate
    farr aPM;    // learning rate component
    farr sPM;    // learning rate component
    farr x1;     // features (0/1) of compound stimulus 1
    farr x2;     // features (0/1) of compound stimulus 2
    farr x;      // features (0/1) of selected compound stimulus
    flt Rew;     // perceived reward
    flt Q1;      // estimated value compound stimulus 1
    flt Q2;      // estimated value compound stimulus 2
    flt Q;       // estimated value of selected compound stimulus
    flt delt;    // prediction error (TD error)
    int choice;  // choice of compound stimulus (1 or 2)
};

template<typename flt, typename mlp>
void PMagent<flt, mlp>::Update_w()
{
    for (int i = 0; i < nsd; ++i) {
        w[i] += alph[i]*x[i]*delt;
    }
}

template<typename flt, typename mlp>
void PMagent<flt, mlp>::Update_alph(const mlp& lp)
{
    flt gam_a = lp.gam_a;
    flt gam_s = lp.gam_s;
    for (int i = 0; i < nsd; ++i) {
        // only update for stimulus components that are 'present'
        if (x[i] > 0) {
            flt adiff1 = std::abs(Rew - w[i]*x[i]);
            flt adiff2 = std::abs(Rew - (Q - w[i]*x[i]));

            aPM[i] += gam_a*(adiff2 - adiff1);
            sPM[i] = (1 - gam_s)*sPM[i] + gam_s*adiff1;

            // impose lower and upper limits on aPM and sPM, mainly following
            // Pearce and Mackintosh (2010)
            if (aPM[i] > 0.1) {
                aPM[i] = 0.1;
            } else if (aPM[i] < 0.01) {
                aPM[i] = 0.01;
            }
            if (sPM[i] > 1) {
                sPM[i] = 1;
            } else if (sPM[i] < 0.05) {
                sPM[i] = 0.05;
            }
            alph[i] = aPM[i]*sPM[i];
        }
    }
}


#endif // AGENT_HPP
