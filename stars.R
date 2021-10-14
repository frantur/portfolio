
rm(list=ls())

setwd('C:/Users/Administrator/Desktop/DATA_Italy_Materiale/Unsupervised Learning/')
stars.df = read.csv('Stars.csv')

summary(stars.df)

stars.std = stars.df[,1:4]
summary(stars.std)

logL = log(stars.std[,2])
logR = log(stars.std[,3])

stars.std[,2] = logL
stars.std[,3] = logR

x = apply(stars.std, 2, median)
y = apply(stars.std, 2, mad)

stars.std = sweep(stars.std, 2, x, FUN = '-')
stars.std = sweep(stars.std, 2, y, FUN = '/')

save(stars.std, file = 'starz.rda')

any(is.na(stars.std))
any(is.null(stars.std))

library(R.utils)
any(isZero(stars.std))

any(is.character(stars.std))

pairs(stars.std)

library(cluster)

stars.pam = pam(stars.std, 9)
fviz_cluster(stars.pam)
summary(stars.pam)

library(factoextra)
fviz_nbclust(stars.std, kmeans, method = 'silhouette', nstart = 100, iter.max = 1000)

stars.kmean = kmeans(stars.std, 6, nstart = 50, iter.max = 1000)

table(stars.df$Type, stars.pam$clustering)
table(stars.df$Type, stars.kmean$cluster)

library(tclust)
stars.curves = ctlcurves(stars.std, k = 4:7, alpha = seq(0, 0.1, len = 5))
plot.ctlcurves(stars.curves)

stars.trim = tkmeans(stars.std, 6, 0.075, 1000)
plot.tclust(stars.trim)

table(stars.df$Type, stars.trim$cluster)
