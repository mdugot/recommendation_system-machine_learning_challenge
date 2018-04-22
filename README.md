# recommendation_system-machine_learning_challenge
Using neural network, analyse items that have been liked by a user and recommend other items to him.  
The neural network is built from scratch in R and can be used in the R console as follows :  

>  \> source("prepareData.R")  
>  \> source("nn.R")  

>  \# parameters : training data, unrolled weigths, labels, learning rate, iterations, batch size, batch number   
>  \> r = learnBatch(training$data, unroll(theta), output$data, 0.5, 1, 100, 100)  
 
>  \> w = roll(r$par, ncol(testing$data), hiddenLayers, ncol(output$data))  
>  \> p = savePrediction(testing$data, w, "result.csv")  

