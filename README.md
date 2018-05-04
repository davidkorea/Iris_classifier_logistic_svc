# Iris_classifier_logistic_svc

# 1. Claasification Model
 
> predict discrete problems

## 1.1 Logistic Regression

* one of the classification models
* comes from LinearREgression, and nonlinearize y = wx+b to below function
* y = 1 / 1 + e^-z , z = wx+b, y∈（0, 1）

    ![](https://www.zhihu.com/equation?tex=Sigmoid%28z%29%3D%5Cfrac%7B1%7D%7B1%2Be%5E%7B-z%7D%7D)
    
    ![](https://pic1.zhimg.com/v2-9bef04e41b7824f6b03e932a72da9e1e_r.jpg)
* if predict y > 0.5 => 1, else y => 0
* params in logistic regression is the same as linear regression, w = coef_, b = intercept_
    
    ![](https://github.com/davidkorea/Iris_classifier_logistic_svc/blob/master/images/logistic.jpg?raw=true)
 
## 1.2 SVM / Support Vector Machine

* Zhihu: [reference](https://www.zhihu.com/question/21094489)

![](https://pic3.zhimg.com/00becdd15361c8e5ceb65da02bcf7fda_r.jpg)

![](https://github.com/davidkorea/Iris_classifier_logistic_svc/blob/master/images/svc.jpg)

![](https://github.com/davidkorea/Iris_classifier_logistic_svc/blob/master/images/svc2.jpg)

 
# 2. Overfitting

![](https://pic3.zhimg.com/80/161572300b52797716017450bcccebb9_hd.jpg)

![](https://github.com/davidkorea/Iris_classifier_logistic_svc/blob/master/images/guonihe.jpg)

# 3. Regulation

![](https://github.com/davidkorea/Iris_classifier_logistic_svc/blob/master/images/zhengze.jpg)

![](https://github.com/davidkorea/Iris_classifier_logistic_svc/blob/master/images/c.jpg)
