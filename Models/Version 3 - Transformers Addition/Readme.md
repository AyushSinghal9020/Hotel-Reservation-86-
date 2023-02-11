# Function Transformers

Machine learning `function transformers` are used to `transform` a set of `features` by applying a specific function to each `feature`. This allows the model to learn non-linear relationships between features and target variables. These transforms can be `user-defined` or `predefined` functions. Common examples of functional transformations are 

* **Log Transformers** - A function that transforms features by applying a `logarithmic function`. It is commonly used to handle right skwed data. This type of transformation helps `normalize` the data and make it more suitable for linear models. Logarithmic transformation is also useful for improving the interpretability of model predictions. 

$$y = f(x) = log(x)$$

* **Square Tranformers** - A type of feature transformation that `squares` the value. Especially useful when the relationship is quadratic in nature. Quadratic transformations are commonly used in `regression` models . However, it is important to note that `adding polynomial` terms to the model can lead to `overfitting` if there is not enough data to support the additional complexity. 

$$y = f(x) = x^n where n = {even}$$

* **Box-Cox Transformers** - A tranformation used to transform skewed data into a normal/Gaussian distribution. This transformation is defined by a `performance parameter` `lambda` used to transform the data. However, the choice of `lambda` parameters can greatly `affect the result` of the transformation, so it is important to carefully consider the lambda choice for best results.

$$ x_i^λ = \displaystyle \Bigg[\frac {\frac {x_i^λ - 1}{λ}}{log (x_i) } \frac {if}{if} \frac {λ != 0}{λ = 0}\Bigg]$$

* **Yeo-Jonhson Transformers** - A `power transformation` method used to handle `positive and negative skewness` in data. It is often used to `preprocess` `non-normally distributed` features to `improve` the `performance` of `models`. A transformation is applied to each feature to make the data more suitable for modeling using Gaussian-based algorithms by tuning the performance parameter that optimizes the normality of the transformed features.

$$x_i^λ = \Bigg[\frac {\frac {[{(x_i +1)}^λ - 1]λ}{log (x_i) + 1}}{\frac {-[{(-x_i + 1)}^{2 - λ}]/2 - λ}{- log(-x_i +1}} \frac {\frac{if}{if}}{\frac {if}{if}} \frac {\frac {y != 0}{λ = 0}}{\frac {λ != 2 ,}{λ = 2}} \frac{\frac {x_i >= 0}{x_i >= 0}}{\frac { x_i < 0}{x_i < 0}}\Bigg]$$
