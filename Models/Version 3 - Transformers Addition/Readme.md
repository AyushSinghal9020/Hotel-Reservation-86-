# Function Transformers
Machine learning function transformers are used to transform a set of features by applying a specific function to each feature. This allows the model to learn non-linear 
relationships between features and target variables. These transforms can be user-defined or predefined functions. Common examples of functional transformations are 
logarithmic functions, polynomial functions, and exponential functions. They are commonly used in feature engineering to improve model performance and handle data 
nonlinearities. 
* **Log Transformers** - A logarithmic (logarithmic) transformer in machine learning is a function that transforms features by applying a logarithmic function. 
Logarithmic transformation is commonly used to handle skewed or skewed data and to reduce the size of outliers. This type of transformation helps normalize the data and 
make it more suitable for linear models. The choice of logarithm base (such as natural logarithm or base 10 logarithm) depends on the distribution of the data and the 
specific needs of the model. Logarithmic transformation is also useful for improving the interpretability of model predictions. 

$$y = f(x) = log(x)$$


* **Square Tranformers** - A squaring transformer in machine learning is a type of feature transformation that squares the value of each feature in the feature set. This 
type of transformation helps capture non-linear relationships between features and target variables. Especially useful when the relationship is quadratic in nature. 
Quadratic transformations are commonly used in regression models, especially to model the relationship between predictors and continuous outcomes. However, it is 
important to note that adding polynomial terms to the model can lead to overfitting if there is not enough data to support the additional complexity. 

$$y = f(x) = x^n where n = {even}$$

* **Box-Cox Transformers** - A Box-Cox transformation is a type of functional transformation used in machine learning to transform non-normal data into a normal 
distribution. This transformation is defined by a performance parameter lambda used to transform the data. The Box-Cox transformation normalizes the data and is 
useful for improving the performance of linear models that assume normality of the target variable. However, the choice of lambda parameters can greatly affect the 
result of the transformation, so it is important to carefully consider the lambda choice for best results.

$$ x_i^λ = \displaystyle \Bigg[\frac {\frac {x_i^λ - 1}{λ}}{log (x_i) } \frac {if}{if} \frac {λ != 0}{λ = 0}\Bigg]$$

* **Yeo-Jonhson Transformers** - The Yeo-Johnson transform is a power transformation method used to handle positive and negative skewness in data. Unlike the common 
Box-Cox transform, the Yeo-Johnson transform can handle both positive and negative values and zero values in your data. In machine learning, it is often used to 
preprocess non-normally distributed features to improve the performance of models such as linear regression. A transformation is applied to each feature to make the 
data more suitable for modeling using Gaussian-based algorithms by tuning the performance parameter that optimizes the normality of the transformed features.

$$x_i^λ = \Bigg[\frac {\frac {[{(x_i +1)}^λ - 1]λ}{log (x_i) + 1}}{\frac {-[{(-x_i + 1)}^{2 - λ}]/2 - λ}{- log(-x_i +1}} \frac {\frac{if}{if}}{\frac {if}{if}} \frac {\frac {y != 0}{λ = 0}}{\frac {λ != 2 ,}{λ = 2}} \frac{\frac {x_i >= 0}{x_i >= 0}}{\frac { x_i < 0}{x_i < 0}}\Bigg]$$
