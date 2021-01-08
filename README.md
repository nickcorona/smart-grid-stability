### Predict titanic survival

I achieved validation accuracy of 82.51%.

I use similarity encoding from the dirty_cat package to encode Name, Ticket, and Cabin numbers. It's an efficient way to deal with high-cardinality categories while maintaining most signal with zero manual preprocessing. Automatic is better if it yields the same results! I kept PassengerId because even though it makes sense for it to have no signal, it turns out that's not true, and it's useful for predicting.

Final feature importance list:
![feature importance](figures/feature_importance.png)
