source("nn.R")
library(data.table)

max = 4712

loadDataFromLabels <- function(filename) {
	labels = fread('input_train.csv', header = T, sep = ";", drop=c("user_id"), showProgress=TRUE)
	labels = lapply(labels, strsplit, split = "/")
	labels = lapply(labels[[1]], strtoi)
	data = matrix(0, nrow=length(labels), ncol=max)

	for (index in seq(length(labels)) ) {
		data[index, labels[[index]] ] = 1
	}

	return (list(labels=labels, data=data))

}

message("load training data...")

training = loadDataFromLabels("input_train.csv")

message("load outputs...")

output = loadDataFromLabels("output.csv")

message("load data to test...")

testing = loadDataFromLabels("input_test.csv")

message("initialize theta...")
hiddenLayers = c(20)
theta = initializeWeigth(training$data, output$data, hiddenLayers)
