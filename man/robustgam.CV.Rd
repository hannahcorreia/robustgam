\name{robustgam.CV}
\alias{robustgam.CV}
\title{
Smoothing parameter selection by robust cross validation
}

\description{
This function combines the \code{\link{robustgam}} with automatic smoothing parameter selection. The smoothing parameter is selected through robust cross validation criterion described in \emph{Wong, Yao and Lee (2012)}. The criterion is designed to be robust to outliers. This function uses grid search to find the smoothing parameter that minimizes the criterion.
}

\usage{
robustgam.CV(X, y, family, p=3, K=30, c=1.345, show.msg=FALSE, count.lim=200,
             w.count.lim=50, smooth.basis="tp", wx=FALSE, sp.min=1e-7, sp.max=1e-3,
             len=50, show.msg.2=TRUE, ngroup=length(y))
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
\item{sp.min}{A vector of minimum values of the searching range for smoothing parameters. If only one value is specified, it will be used for all smoothing parameters.}
\item{sp.max}{A vector of maximum values of the searching range for smoothing parameters. If only one value is specified, it will be used for all smoothing parameters.}
\item{len}{A vector of grid sizes. If only one value is specified, it will be used for all smoothing parameters.}
\item{show.msg.2}{If \code{show.msg.2=T}, progress of the grid search is displayed.}
\item{ngroup}{number of group used in the cross validation. If \code{ngroup=length(y)}, full cross validation is implemented. If \code{ngroup=2}, two-fold cross validation is implemented.}
}

\value{
\item{fitted.values}{fitted values (of the optimum fit)}
\item{initial.fitted}{the starting values of the algorithm (of the optimum fit)}
\item{beta}{estimated coefficients (corresponding to the basis) (of the optimum fit)}
\item{optim.index}{the index of the optimum fit}
\item{optim.index2}{the index of the optimum fit in another representation:

\code{optim.ndex2=arrayInd(optim.index,len)}}
\item{optim.criterion}{the optimum value of robust cross validation criterion}
\item{optim.sp}{the optimum value of the smoothing parameter}
\item{criteria}{the values of criteria for all fits during grid search}
\item{sp}{the grid of smoothing parameter}
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
robustfit.gic <- robustgam.CV(x, y, family=true.family, p=3, c=1.6, show.msg=FALSE,
  count.lim=400, smooth.basis='tp',ngroup=5); robustfit <- robustfit.gic$optim.fit


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
