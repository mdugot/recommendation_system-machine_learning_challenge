activation <- function(x) {
	r = (1 / (1 + exp(-x)))
	return (r)
}

derivative <- function(n) {
	return (n * (1 - n))
}

outputActivation <- function(x) {
	r = (1 / (1 + exp(-x)))
	return (r)
}


correctRate <- function(x, w, o, hiddenLayers, lambda) {
	message("forward propagation")
	r = forwardPropagation(x, roll(w, ncol(x), hiddenLayers, ncol(o)) )
	message("convert result to recommendations")
	recommend =  t(apply(r$hypothesis, 1, order, decreasing=T))[,1:5]
	check = t(sapply(1:nrow(o),   function(i, m, sm) {return (m[i, sm[i,] ] )}   , m=o, sm=recommend))

	message("cost : ", cost(w, x, o, lambda, hiddenLayers))
	message("correct : ", sum(check), "/", nrow(o) * 5)
	message("rate : ", round(sum(check) / (nrow(o) * 5) * 100), "%")
}

savePrediction <- function(data, weigth, filename) {

	prediction = forwardPropagation(data, weigth)
	recommend =  t(apply(prediction$hypothesis, 1, order, decreasing=T))[,1:5]
	srec = sapply(1:nrow(recommend), function(index, data) {paste(data[index,], collapse="/")}, data=recommend)
	srec = cbind(10001:(10000+length(srec)), matrix(srec, ncol=1))
	colnames(srec) = c("user_id", "items")
	write.csv2(srec, file = filename, sep=";", row.names = FALSE, quote = FALSE)

	return (srec)

}

randomMatrix <- function(w, h, e) {
	tmp = runif(w*h, -e, e)
	return (matrix(tmp, ncol=w, nrow=h))
}

initializeWeigth <- function(trainingData, output, hiddenLayers, epsilon = 0.1) {
	layers = c(ncol(trainingData), hiddenLayers, ncol(output))
	weigth = list()
	for (i in 2:length(layers)) {
		weigth[[i - 1]] = randomMatrix(layers[i], layers[i - 1] + 1, epsilon)
	}
	return (weigth)
}

forwardPropagation <- function(input, weigth) {
	tmp = input
	result = list()
	if (is.vector(tmp))
		tmp = matrix(tmp, nrow=1)
	result[[1]] = tmp
	for (i in 1:(length(weigth) - 1)) {
		tmp = activation(cbind(1,tmp) %*% weigth[[i]])
		result[[i + 1]] = tmp
	}
	i = length(weigth)
	tmp = outputActivation(cbind(1,tmp) %*% weigth[[i]])
	result[[i + 1]] = tmp
	return (list(hypothesis=tmp, propagation=result))
}

wsum <- function(w) {
	r = 0
	for (i in 1:length(w)) {
		r = r + sum(w[[i]][-1,] ^ 2)
	}
	return (r)
}


initializeDelta <- function(weigth) {
	delta = weigth
	for (i in 1:length(delta))
		delta[[i]][] = 0
	return (delta)
}

unroll <- function(datalist) {
	result = vector()
	for (m in datalist) {
		result = c(result, as.vector(m))
	}
	return (result)
}

unrollLength <- function(datalist) {
	result = 0
	for (m in datalist) {
		result = result + length(m)
	}
	return (result)
}

roll <- function(data, inputLayer, hiddenLayers, outputLayer) {
	result = list()
	layers = c(inputLayer, hiddenLayers, outputLayer)

	start = 1

	for (i in 2:length(layers)) {
		
		end = start + (layers[i - 1]+1) * layers[i] - 1

		result[[i - 1]] = matrix(data[start:end], nrow = layers[i -1] + 1, ncol = layers[i])

		start = end + 1
	}
	return (result)
}

cost <- function(par, data, y, lambda, hiddenLayers) {
	weigth = roll(par, ncol(data), hiddenLayers, ncol(y))
	m = nrow(data)
	h = forwardPropagation(data, weigth)$hypothesis
	return ( (-1/m) * sum(y * log(h) + (1 - y) * log(1 - h)) + ( (lambda/(2*m)) * wsum(weigth) ) )
}

checkGradient <- function(par, data, y, lamda, hiddenLayers) {
	epsilon = 0.1
	D = vector("double", length(par))
	progress = txtProgressBar(min = 1, max = length(D), initial = 1, style = 3)
	for (i in 1:length(D)) {
		setTxtProgressBar(progress, i)
		w = par[[i]]

		par[[i]] = w - epsilon
		bc = cost(par, data, y, lamda, hiddenLayers)

		par[[i]] = w + epsilon
		ac = cost(par, data, y, lamda, hiddenLayers)
		
		D[[i]] = (ac - bc) / (2 * epsilon)

		par[[i]] = w
	}

	return (D)
}


gradient <- function(par, data, y, lambda, hiddenLayers) {
	weigth = roll(par, ncol(data), hiddenLayers, ncol(y))
	m = nrow(data)
	error = list()
	delta = initializeDelta(weigth)
	for (i in 1:m) {
		f = forwardPropagation(data[i,], weigth)
		error[[length(weigth) + 1]] = t(f$hypothesis - y[i,])
		for (j in (length(weigth)):1) {
			error[[j]] = (weigth[[j]][-1,] %*% error[[j+1]] ) * t(derivative(f$propagation[[j]]))
			delta[[j]] = delta[[j]] + t(error[[j+1]] %*% cbind(1,f$propagation[[j]]))
		}
	}
	D = vector("double", length(par))
	start = 1
	for (i in 1:length(delta)) {
		
		end = start + length(delta[[i]])
		biaisEnd = start + ncol(delta[[i]])
		
		Db = (delta[[i]][1,] / m)
		Dw = (delta[[i]][-1,] / m) + ( (lambda / m) * weigth[[i]][-1,] )
		if (is.matrix(Dw)) {
			D[start:(end-1)] = as.vector( rbind(Db, Dw)  )
		} else {
			D[start:(end-1)] = as.vector( c(Db, Dw)  )
		}
		start = end
	}
	return (D)
}

learnBatch <-function(data, theta, y, lambda, iterations, batch, batchIterations) {
	p = theta
	progress = txtProgressBar(min = 1, max = batchIterations, initial = 1, style = 3)
	for (i in 1:batchIterations) {
		setTxtProgressBar(progress, i)
		index = sample(1:nrow(data), batch)
		r = learn(data[index,,drop=F], p, y[index,,drop=F], lambda, iterations)
		p = r$par
	}
	return (r)
}

learn <- function(data, theta, y, lambda, iterations) {
	optim(par=theta, fn=cost, gr=gradient, data=data, y=y, lambda=lambda, hiddenLayers=hiddenLayers, method = "L-BFGS-B", control=list(trace=0, maxit=iterations, REPORT=1))
}
