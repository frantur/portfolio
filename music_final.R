
### ----- Preliminary operations ----- ###

rm(list=ls())

#set work directory
getwd()
setwd('C:/Users/Administrator/Desktop/DATA_Italy_Materiale/Unsupervised Learning/music_feats/')

#import and merge datasets
df.temp1 = read.csv('data.csv')
df.temp2 = read.csv('data_2genre.csv')

music.df = rbind(df.temp1, df.temp2)

rm(df.temp1); rm (df.temp2)

#summary and unique values of dataframe
summary(music.df)
unique(music.df$label)

#checking for missing values in dataset
any(is.na(music.df))
any(is.null(music.df))

#checking for zero values in dataset
library(R.utils)
any(isZero(music.df[,-c(1,30)]))

#checking for duplicated entries
any(duplicated(music.df))
which(duplicated(music.df))

#so it is found that the second imported dataset it's a replica
#of some of the rows of the first dataset
#it will be used only the first imported dataset
rm(music.df)
music.df = read.csv('data.csv')

#for consistency, the same cleaning operations will be replied

#summary and unique values of dataframe
summary(music.df)
unique(music.df$label)

#checking for missing values in dataset
any(is.na(music.df))
any(is.null(music.df))

#checking for zero values in dataset
library(R.utils)
any(isZero(music.df[,-c(1,30)]))

#checking for duplicated entries
any(duplicated(music.df))

#export cleaned dataset as R object for reproducibility
save(music.df, file = 'music.df.rda')

### ----- Standardization ----- ###

#numeric variables selection
music.std = music.df[,2:29]

#median and mad
x = apply(music.std, 2, median)
y = apply(music.std, 2, mad)

#centering around median and standardization
music.std = sweep(music.std, 2, x, FUN = '-')
music.std = sweep(music.std, 2, y, FUN = '/')

#visualization of predictors for getting an idea
#correlation panel for pairs plot
panel.cor <- function(a, b){
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  r <- round(cor(a, b), digits=2)
  txt <- paste0("R = ", r)
  cex.cor <- 0.8/strwidth(txt)
  text(0.5, 0.5, txt, cex = cex.cor)# * r)
}
#pairs plot with correlations for arbitrary features subsets
pairs(music.std[,1:8], lower.panel = panel.cor)
pairs(music.std[,9:16], lower.panel = panel.cor)

### ----- K-means Analysis ----- ###

#computing the optimal number of clusters (and verifying if a group structure exists)
library(factoextra)
fviz_nbclust(music.std, kmeans, method = 'wss', nstart = 100, iter.max = 1000)
fviz_nbclust(music.std, kmeans, method = 'silhouette', nstart = 100, iter.max = 1000)

#clustering with K-means
music.kmean = kmeans(music.std,3,nstart=100,iter.max=1000)

#visualization of clustering
music.kmean

#examples of plots
plot(music.std$mfcc3, music.std$spectral_centroid, col = music.kmean$cluster)
#adding centers to the previous plot
points(music.kmean$centers[,11], music.kmean$centers[,5], pch = 7, cex = 1.5, col = 'blue')

plot(music.std$mfcc10, music.std$spectral_centroid, col = music.kmean$cluster)
points(music.kmean$centers[,18], music.kmean$centers[,5], pch = 7, cex = 1.5, col = 'blue')

#table comparison with a priori labels
table(music.df$label, music.kmean$cluster)


### ----- PAM Analysis ----- ###

#clustering with the PAM method
library(cluster)
music.pam = pam(music.std, 3)

summary(music.pam)

#cluster visualization
fviz_cluster(music.pam, data=music.std)

#table comparison with a priori labels
table(music.df$label, music.pam$clustering)


### ----- Trimming ----- ###

#estimation of level of trimming
library(tclust)
music.curves = ctlcurves(music.std, k=1:5,alpha = seq(0, 0.25, len = 5))
plot.ctlcurves(music.curves)

#fine tuning for the trimming parameter
music.curves = ctlcurves(music.std, k=3,alpha = seq(0, 0.1, len = 5))
plot.ctlcurves(music.curves)

#trimmed K-means
music.trim = tkmeans(music.std, k=3, alpha=0.025, iter.max = 1000)

#plotting the trimmed clustering
plot.tclust(music.trim)

#table comparison with a priori labels
table(music.df$label, music.trim$cluster)

#swamping test
idx.out = which(music.trim$cluster == 0)
dist.mat = matrix(NA, nrow = length(idx.out), ncol = ncol(music.trim$centers))

for(i in 1:nrow(dist.mat)){
  for(k in 1:ncol(dist.mat)){
    dist.mat[i,k] = sum((music.std[idx.out[i],]-music.trim$centers[,k])^2)
  }
}

dist.clust = apply(dist.mat, 1, min)
out.test = dist.clust > qchisq(.975, ncol(music.std))

#FALSE equals swamped value
out.test

#number of swamped values
length(which(out.test == FALSE))


### ----- Principal Component Analysis ----- ###

#principal components
music.pca = prcomp(music.std)
plot(music.pca, type = 'l')
summary(music.pca)

#graphical visulization of the contributions
fviz_eig(music.pca)
fviz_contrib(music.pca, choice = 'var', axes = 6)

#using first 6 components (~80% of total variance) for analysis
music.redux = data.frame(music.pca$x[,1:6])

#computing the optimal number of clusters (and verifying if a group structure exists)
fviz_nbclust(music.redux, kmeans, method = 'wss', nstart = 100, iter.max = 1000)
fviz_nbclust(music.redux, kmeans, method = 'silhouette', nstart = 100, iter.max = 1000)

#and then we repeat the same k-means analysis with 6 components

#clustering with K-means
music.redux.kmean = kmeans(music.std,3,nstart=100,iter.max=1000)

#visualization of clustering (useful for plotting)
music.redux.kmean

#examples of plots
plot(music.redux$PC1, music.redux$PC2, col = music.redux.kmean$cluster)
plot(music.redux$PC2, music.redux$PC3, col = music.redux.kmean$cluster)
plot(music.redux$PC1, music.redux$PC3, col = music.redux.kmean$cluster)

#table for cross reference
table(music.df$label, music.redux.kmean$cluster)

#estimation of level of trimming
music.redux.curves = ctlcurves(music.std, k=1:5,alpha = seq(0, 0.25, len = 5))
plot.ctlcurves(music.redux.curves)

#fine tuning for the trimming parameter
music.redux.curves = ctlcurves(music.std, k=3,alpha = seq(0, 0.1, len = 5))
plot.ctlcurves(music.redux.curves)

#trimmed K-means
library(tclust)
music.redux.trim = tkmeans(music.std, k=3, alpha=0.025, iter.max = 1000)

#plotting the trimmed clustering
plot.tclust(music.redux.trim)

#table comparison with a priori labels
table(music.df$label, music.redux.trim$cluster)

#swamping test
idx.out.redux = which(music.redux.trim$cluster == 0)
dist.mat.redux = matrix(NA, nrow = length(idx.out.redux), ncol = ncol(music.redux.trim$centers))

for(i in 1:nrow(dist.mat.redux)){
  for(k in 1:ncol(dist.mat.redux)){
    dist.mat.redux[i,k] = sum((music.redux[idx.out.redux[i],]-music.redux.trim$centers[,k])^2)
  }
}

dist.clust.redux = apply(dist.mat.redux, 1, min)
out.test.redux = dist.clust.redux > qchisq(.975, ncol(music.redux))

#FALSE equals swamped value
out.test.redux

#number of swamped values
length(which(out.test.redux == FALSE)) 

### ----- Exporting Plots and Tables ----- ###

#this is an example of the code used for exporting images
library(gridExtra)
png('pca_trim_tab.png')
grid.table(table(music.df$label, music.redux.trim$cluster))
dev.off()

png('pca_trim_clus.png')
plot.tclust(music.redux.trim)
dev.off()
