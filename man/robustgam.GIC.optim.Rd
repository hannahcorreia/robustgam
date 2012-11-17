\name{robustgam.GIC.optim}
\alias{robustgam.GIC.optim}
\title{
Smoothing parameter selection by GIC (by optim)
}

\description{
This function is the same as \code{\link{robustgam.GIC}}, except that the R internal optimization function \code{\link{optim}} is used to find the smoothing parameter that minimizes the RAIC or RBIC criterion.
}

\usage{
robustgam.GIC.optim(X, y, family, p=3, K=30, c=1.345, show.msg=FALSE, count.lim=200,
                    w.count.lim=50, smooth.basis="tp", wx=FALSE, lsp.initial=log(1e-4),
                    lsp.min=-20, lsp.max=10, gic.constant=log(length(y)),
                    method="L-BFGS-B",optim.control=list(trace=1))
}

\arguments{
\item{X}{a vector or a matrix (each covariate form a column) of covariates}
\item{y}{a vector of responses}
\item{family}{A family object specifying the distribution and the link function. See \code{\link{glm}} and \code{\link{family}}.}
\item{p}{order of the basis. It depends on the option of smooth.basis.}
\item{K}{number of knots of the basis; dependent on the option of smooth.basis.}
\item{c}{tunning parameter for Huber function; a smaller value of c corresponds to a more robust fit. It is recommended to set as 1.2 and 1.6 for binomial and poisson distribution respectively.}
\item{show.msg}{If \code{show.msg=T}, progress of robustgam is displayed.}
\item{count.lim}{maximum number of iterations of the whole algorithm}
\item{w.count.lim}{maximum number of updates on the weight. It corresponds to zeta in \emph{Wong, Yao and Lee (2012)}}
\item{smooth.basis}{the specification of basis. Four choices are available: \code{"tp"} = thin plate regression spline, \code{"cr"} = cubic regression spline, \code{"ps"} = P-splines, \code{"tr"} = truncated power spline. For more details, see \code{\link{smooth.construct}}.}
\item{wx}{If \code{wx=T}, robust weight on the covariates are applied. For details, see Real Data Example in \emph{Wong, Yao and Lee (2012)}}
\item{lsp.initial}{A vector of initial values of the \emph{log} of smoothing parameters used to start the optimization algorithm.}
\item{lsp.min}{a vector of minimum values of the searching range for the \emph{log} of smoothing parameters. If only one value is specified, it will be used for all smoothing parameters.}
\item{lsp.max}{a vector of maximum values of the searching range for the \emph{log} of smoothing parameters. If only one value is specified, it will be used for all smoothing parameters.}
\item{gic.constant}{If \code{gic.contant=log(length(y))}, robust BIC is used. If \code{gic.constant=2}, robust AIC is used.}
\item{method}{method of optimization. For more details, see \code{\link{optim}}}
\item{optim.control}{setting for \code{\link{optim}}. For more details, see \code{\link{optim}}}
}

\value{
\item{fitted.values}{fitted values (of the optimum fit)}
\item{beta}{estimated coefficients (corresponding to the basis) (of the optimum fit)}
\item{beta.fit}{for internal use}
\item{gic}{the optimum value of robust AIC or robust BIC}
\item{sp}{the optimum value of the smoothing parameter}
\item{gic.optim}{the output of \code{\link{optim}}}
\item{w}{for internal use}
\item{gic.constant}{the \code{gic.constant} specified in the input}
\item{optim.fit}{the robustgam fit object of the optimum fit. It is handy for applying the prediction method.}
}

\references{
Raymond K. W. Wong, Fang Yao and Thomas C. M. Lee (2012) \emph{Robust Estimation for Generalized Additive Models}. Submitted to the \emph{Journal of Graphical and Computational Statistics}.
}

\author{
Raymond K. W. Wong <raymondkww.dev@gmail.com>
}

\seealso{
\code{\link{robustgam.GIC}}, \code{\link{robustgam.GIC.optim}}, \code{\link{robustgam.CV}}, \code{\link{pred.robustgam}}
}

\examples{
# load library
library(robustgam)

# test function
test.fun <- function(x, ...) {
  return(2*sin(2*pi*(1-x)^2))
}

# some setting
set.seed(1234)
true.family <- poisson()
out.prop <- 0.05
n <- 100

# generating dataset for poisson case
x <- runif(n)
x <- x[order(x)]
true.eta <- test.fun(x)
true.mu <- true.family$linkinv(test.fun(x))
y <- rpois(n, true.mu) # for poisson case

# create outlier for poisson case
out.n <- trunc(n*out.prop)
out.list <- sample(1:n, out.n, replace=FALSE)
y[out.list] <- round(y[out.list]*runif(out.n,min=3,max=5)^(sample(c(-1,1),out.n,TRUE)))

\dontrun{

# robust GAM fit
robustfit.gic <- robustgam.GIC.optim(x, y, family=true.family, p=3, c=1.6, show.msg=FALSE,
  count.lim=400, smooth.basis='tp', lsp.initial=log(1e-2) ,lsp.min=-15, lsp.max=10,
  gic.constant=log(n), method="L-BFGS-B"); robustfit <- robustfit.gic$optim.fit


# ordinary GAM fit
nonrobustfit <- gam(y~s(x, bs="tp", m=3),family=true.family) # m = p for 'tp'

# prediction
x.new <- seq(range(x)[1], range(x)[2], len=1000)
robustfit.new <- pred.robustgam(robustfit, data.frame(X=x.new))$predict.values
nonrobustfit.new <- as.vector(predict.gam(nonrobustfit,data.frame(x=x.new),type="response"))

# plot
plot(x, y)
lines(x.new, true.family$linkinv(test.fun(x.new)), col="blue")
lines(x.new, robustfit.new, col="red")
lines(x.new, nonrobustfit.new, col="green")
legend(0.6, 23, c("true mu", "robust fit", "nonrobust fit"), col=c("blue","red","green"),
  lty=c(1,1,1))

}
}
