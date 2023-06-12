# functions to read data from hdf5 file created by simulation program

# return data table for "one-column" phenotype data
h5_dt <- function(hf_name) {
    require(hdf5r)
    require(data.table)
    f.h5 <- H5File$new(hf_name, mode = "r")
    atyp <- f.h5[["atyp"]][]
    repl <- f.h5[["repl"]][]
    tstep <- f.h5[["tstep"]][]
    choice <- f.h5[["choice"]][]
    cstyp1 <- f.h5[["cstyp1"]][]
    cstyp2 <- f.h5[["cstyp2"]][]
    Rew <- f.h5[["Rew"]][]
    Q1 <- f.h5[["Q1"]][]
    Q2 <- f.h5[["Q2"]][]
    Q <- f.h5[["Q"]][]
    Q1tr <- f.h5[["Q1tr"]][]
    Q2tr <- f.h5[["Q2tr"]][]
    delt <- f.h5[["delt"]][]
    f.h5$close_all()
    data.table(atyp = atyp, repl = repl, tstep = tstep, 
               choice = choice, cstyp1 = cstyp1, cstyp2 = cstyp2,
               Rew = Rew, Q1 = Q1, Q2 = Q2, Q = Q, 
               Q1tr = Q1tr, Q2tr = Q2tr, delt = delt)
}

# return matrix where each row is a feature vector x1
h5_x1 <- function(hf_name) {
    require(hdf5r)
    f.h5 <- H5File$new(hf_name, mode = "r")
    x1 <- t(f.h5[["x1"]][,])
    f.h5$close_all()
    x1
}

# return matrix where each row is a feature vector x2
h5_x2 <- function(hf_name) {
    require(hdf5r)
    f.h5 <- H5File$new(hf_name, mode = "r")
    x2 <- t(f.h5[["x2"]][,])
    f.h5$close_all()
    x2
}

# return matrix where each row is a feature vector x2
h5_x <- function(hf_name) {
    require(hdf5r)
    f.h5 <- H5File$new(hf_name, mode = "r")
    x <- t(f.h5[["x"]][,])
    f.h5$close_all()
    x
}

# return vector of true values
h5_w_tr <- function(hf_name) {
    require(hdf5r)
    f.h5 <- H5File$new(hf_name, mode = "r")
    w_tr <- t(f.h5[["w_tr"]][,])
    f.h5$close_all()
    w_tr
}

# return vector of estimated values
h5_w <- function(hf_name) {
    require(hdf5r)
    f.h5 <- H5File$new(hf_name, mode = "r")
    w <- t(f.h5[["w"]][,])
    f.h5$close_all()
    w
}

# return vector of learning rates
h5_alph <- function(hf_name) {
    require(hdf5r)
    f.h5 <- H5File$new(hf_name, mode = "r")
    alph <- t(f.h5[["alph"]][,])
    f.h5$close_all()
    alph
}
