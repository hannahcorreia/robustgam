
// includes from the plugin
#include <RcppArmadillo.h>
#include <Rcpp.h>


#ifndef BEGIN_RCPP
#define BEGIN_RCPP
#endif

#ifndef END_RCPP
#define END_RCPP
#endif

using namespace Rcpp;


// user includes


// declarations
extern "C" {
SEXP rgam_fast_main_cpp( SEXP Ry, SEXP RX, SEXP Rfamily, SEXP Rc, SEXP RB, SEXP RrS, SEXP Rexpect, SEXP Rw_fun, SEXP Rm_initial, SEXP Rbeta_old, SEXP Rcount_lim, SEXP Rw_count_lim, SEXP Rdisplay) ;
}

// definition

SEXP rgam_fast_main_cpp( SEXP Ry, SEXP RX, SEXP Rfamily, SEXP Rc, SEXP RB, SEXP RrS, SEXP Rexpect, SEXP Rw_fun, SEXP Rm_initial, SEXP Rbeta_old, SEXP Rcount_lim, SEXP Rw_count_lim, SEXP Rdisplay ){
BEGIN_RCPP
// rgam.fast.main

// Ry, RX, Rfamily, Rc, RB, RrS, Rexpect, Rw_fun, Rm_initial, Rbeta_old, Rcount_lim, Rw_count_lim, Rdisplay

double c = (Rcpp::as<Rcpp::NumericVector>(Rc))[0];
int count_lim = (Rcpp::as<Rcpp::IntegerVector>(Rcount_lim))[0];
int w_count_lim = (Rcpp::as<Rcpp::IntegerVector>(Rw_count_lim))[0];
int display = (Rcpp::as<Rcpp::IntegerVector>(Rdisplay))[0];
arma::colvec y = Rcpp::as<arma::colvec>(Ry);
Rcpp::NumericMatrix X = Rcpp::as<Rcpp::NumericMatrix>(RX);
arma::colvec m_initial = Rcpp::as<arma::colvec>(Rm_initial);
arma::colvec beta_old = Rcpp::as<arma::colvec>(Rbeta_old);
Rcpp::Function w_fun(Rcpp::as<Rcpp::Function>(Rw_fun));
Rcpp::Function expect(Rcpp::as<Rcpp::Function>(Rexpect));
arma::mat B = Rcpp::as<arma::mat>(RB);
arma::mat rS = Rcpp::as<arma::mat>(RrS);

Rcpp::List family(Rfamily);
Rcpp::Function linkfun(Rcpp::as<Rcpp::Function>(family["linkfun"]));
Rcpp::Function linkinv(Rcpp::as<Rcpp::Function>(family["linkinv"]));
Rcpp::Function mu_eta(Rcpp::as<Rcpp::Function>(family["mu.eta"]));
Rcpp::Function variance(Rcpp::as<Rcpp::Function>(family["variance"]));

int q, n;
q = B.n_cols;
n = B.n_rows;

arma::mat X1 = arma::join_cols(B,rS);

arma::colvec w(n);
arma::colvec m(n);
arma::colvec m_old(m_initial);
arma::colvec weight(n);
arma::colvec z(n);
arma::colvec z_z(n);
arma::colvec mu_eta_val(n);
arma::colvec variance_val(n);
arma::colvec eta(n);
arma::colvec r(n);
arma::colvec huber(n);
arma::colvec expect_val(n);
arma::colvec sqrtVar(n);
arma::colvec am_y(n+q);
arma::colvec am_weight(n+q);
arma::colvec b(q);
double diff_mu = 1;
double diff_beta = 1;
int display_count = 1;

w = Rcpp::as<arma::colvec>(w_fun(Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(m_initial)),Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(c)),X));

while ((diff_beta>1e-7)&&(display_count<=count_lim)){
    //pseduo data
    variance_val = Rcpp::as<arma::colvec>(variance(Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(m_old))));
    sqrtVar = arma::sqrt(variance_val);
    r = (y-m_old)/sqrtVar;
    huber = ((-c)<=r)%(r<=c)%r + (r>c)%(c*arma::ones(n))-(r<(-c))%(c*arma::ones(n));
    expect_val = Rcpp::as<arma::colvec>(expect(Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(m_old)),Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(c)),Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(sqrtVar))));
    z = (huber - expect_val)%w%sqrtVar+m_old;
    //
    eta =  Rcpp::as<arma::colvec>(linkfun(Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(m_old))));
    mu_eta_val = Rcpp::as<arma::colvec>(mu_eta(Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(eta))));
    weight = mu_eta_val/sqrtVar;
    z_z = (z-m_old)/mu_eta_val + eta;
    //fit additve models
    am_y = join_cols(z_z,arma::zeros(q));
    am_weight = join_cols(weight,arma::ones(q));
    b = arma::solve(arma::diagmat(am_weight)*X1,am_weight%am_y);
    m = Rcpp::as<arma::colvec>(linkinv(Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(B*b))));
    //
    diff_mu = arma::norm(m_old-m,2)/std::sqrt(n);
    diff_beta = arma::norm(beta_old-b,2)/std::sqrt(q);
    m_old = m;
    beta_old = b;
    if (display_count <= w_count_lim){
	w = Rcpp::as<arma::colvec>(w_fun(Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(m)),Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(c)),X));
    }
    if (display==1){
	Rcpp::Rcout << "Fit " << display_count << " completed and the difference of beta is "<< diff_beta << ", while that of mu is " << diff_mu << "\n";
    }
    display_count = display_count + 1;
}

int converge = 1;
if (display_count>count_lim){
    converge = 0;
}

return Rcpp::List::create(Rcpp::Named("fitted.values")=m_old,Rcpp::Named("initial.fitted")=m_initial,Rcpp::Named("beta")=beta_old,Rcpp::Named("w")=w,Rcpp::Named("converge")=converge);


END_RCPP
}



