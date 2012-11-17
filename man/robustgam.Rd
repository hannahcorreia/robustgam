\name{robustgam}
\alias{robustgam}
\title{
Pseudo Data Algorithm for Robust Estmation of GAMs
}

\description{
This function implements a fast and stable algorithm developed in \emph{Wong, Yao and Lee (2012)} for robust estimation of generalized additive models. Currently, this implementation only covers binomial and poisson distributions. This function does not choose smoothing parameter by itself and the user has to specify it manunally. For implementation with automatic smoothing parameter selection, see \code{\link{robustgam.GIC}}, \code{\link{robustgam.GIC.optim}}, \code{\link{robustgam.CV}}. For prediction, see \code{\link{pred.robustgam}}.
}

\usage{
robustgam(X, y, family, p=3, K=30, c=1.345, sp=-1, show.msg=FALSE, count.lim=200,
          w.count.lim=50, smooth.basis="tp", wx=FALSE)
}

\arguments{
\item{X}{a vector or a matrix (each covariate form a column) of covariates}
\item{y}{a vector of responses}
\item{family}{A family object specifying the distribution and the link function. See \code{\link{glm}} and \code{\link{family}}.}
\item{p}{order of the basis. It depends on the option of smooth.basis.}
\item{K}{number of knots of the basis; dependent on the option of smooth.basis.}
\item{c}{tunning parameter for Huber function; a smaller value of c corresponds to a more robust fit. It is recommended to set as 1.2 and 1.6 for binomial and poisson distribution respectively.}
\item{sp}{a vector of smoothing parameter. If only one value is specified, it will be used for all smoothing parameters.}
\item{show.msg}{If \code{show.msg=T}, progress is displayed.}
\item{count.lim}{maximum number of iterations of the whole algorithm}
\item{w.count.lim}{maximum number of updates on the weight. It corresponds to zeta in \emph{Wong, Yao and Lee (2012)}}
\item{smooth.basis}{the specification of basis. Four choices are available: \code{"tp"} = thin plate regression spline, \code{"cr"} = cubic regression spline, \code{"ps"} = P-splines, \code{"tr"} = truncated power spline. For more details, see \code{\link{smooth.construct}}.}
\item{wx}{If \code{wx=T}, robust weight on the covariates are applied. For details, see Real Data Example in \emph{Wong, Yao and Lee (2012)}}
}

\value{
\item{fitted.values}{fitted values}
\item{initial.fitted}{the starting values of the algorithm}
\item{beta}{estimated coefficients (corresponding to the basis)}
\item{B}{the basis: fitted linear estimator = \code{B\%\*\%beta}}
\item{sD}{for internal use}
\item{basis}{the smooth construct object. For more details, see \code{\link{smooth.construct}}}
\item{converge}{If \code{converge=T}, the algorithm converged.}
\item{w}{for internal use}
\item{family}{the family object}
\item{wx}{Indicate whether robust weight on covariates is applied}
\item{beta.fit}{for internal use}
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

# robust GAM fit
robustfit <- robustgam(x, y, family=true.family, p=3, c=1.6, sp=0.000143514, show.msg=FALSE,
 smooth.basis='tp')
## the smoothing parameter is selected by the RBIC, the command is:
# robustfit.gic <- robustgam.GIC.optim(x, y, family=true.family, p=3, c=1.6, show.msg=FALSE,
#   count.lim=400, smooth.basis='tp', lsp.initial=log(1e-2) ,lsp.min=-15, lsp.max=10,
#   gic.constant=log(n), method="L-BFGS-B"); robustfit <- robustfit.gic$optim.fit


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
