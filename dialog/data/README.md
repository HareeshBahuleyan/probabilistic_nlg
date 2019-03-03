## Dialog Datasets
For dialog generation experiments, the code can be run with either of the two datasets:
1. Cornell Movie Dialogs
2. DailyDialog

**Note**: The [DialyDialog](https://arxiv.org/pdf/1710.03957.pdf) dataset was downloaded from [here](http://yanran.li/dailydialog.html). However, we found data to be duplicated between train and test sets, i.e., utterance-reply pairs present in the training set were also present in the dev/test sets. We have removed such duplicates before running our models. The de-duplicated dataset is made available for further research ([`DailyDial/de_duplicated`](https://github.com/HareeshBahuleyan/probabilistic_nlg/tree/master/dialog/data/DailyDial/de_duplicated)).