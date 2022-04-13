import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc


def StageNormalize(stage):
    stage = str(stage)
    if re.search('Stage IV', stage):
        return 4
    elif re.search('Stage III', stage):
        return 3
    elif re.search('Stage II', stage):
        return 2
    elif re.search('Stage I', stage):
        return 1
    else:
        return np.nan


class ClfSet():
    def __init__(self, X, y):
        self.X = StandardScaler().fit_transform(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state = 5)

    def GBC(self):
        GBC = GradientBoostingClassifier(learning_rate=0.1, random_state=5, min_samples_leaf=5)
        grid_search_model = GridSearchCV(GBC, {'n_estimators': [50, 100, 150], 'max_depth': [1, 2, 3]}, cv = 4)
        grid_search_model.fit(self.X_train, self.y_train)
        best_GBC = grid_search_model.best_estimator_
        best_GBC.fit(self.X_train, self.y_train)
        y_pred_prob = best_GBC.predict_proba(self.X_test)
        fpr, tpr, threholds = roc_curve(self.y_test, y_pred_prob[:, 1])
        print(auc(fpr, tpr))
        
        # y_pred_prob = GBC.predict_proba(self.X_test)
        # fpr, tpr, threholds = roc_curve(self.y_test, y_pred_prob[:, 1])
        # return auc(fpr, tpr)

    def RFC(self):
        RFC = RandomForestClassifier()
        RFC = RFC.fit(self.X_train, self.y_train)
        y_pred_prob = RFC.predict_proba(self.X_test)
        fpr, tpr, threholds = roc_curve(self.y_test, y_pred_prob[:, 1])
        return auc(fpr, tpr)

    def GBCParaSelection(self):
        kf = KFold(n_splits=4)
        maxAuc, bestTreeNum, bestDepth = 0, 0, 0
        for train_index, test_index in kf.split(self.X_train):
            foldSet = ClfSet(self.X_train, self.y_train)
            foldSet.X_train, foldSet.X_test = self.X_train[train_index], self.X_train[test_index]
            foldSet.y_train, foldSet.y_test = self.y_train[train_index], self.y_train[test_index]
            for treenum in [50, 100, 150]:
                for depth in [1, 2, 3]:
                    if foldSet.GBC(numOfTrees=treenum, maxDepth=depth) > maxAuc:
                        maxAuc = foldSet.GBC(numOfTrees=treenum, maxDepth=depth)
                        bestTreeNum = treenum
                        bestDepth = depth
        return self.GBC(numOfTrees=bestTreeNum, maxDepth=bestDepth)

if __name__ == '__main__':
    # Loading data
    # abundanceMicroFiles =  {'FD': 'microbiomes_Kraken_FULL_abundance.csv',
    #                         'LCR': 'microbiomes_Kraken_LIKELYREMOVE_abundance.csv',
    #                         'APCR': 'microbiomes_Kraken_ALLREMOVE_abundance.csv',
    #                         'PCCR': 'microbiomes_Kraken_PC_abundance.csv',
    #                         'MSF': 'microbiomes_Kraken_MOSTSTRINGENT_abundance.csv'}
    # abundance_trans = pd.read_csv('./selected_transcriptome.csv', index_col = 0)
    # labelsMicroSer = pd.read_csv('../COAD/microbiomes/metadata.csv', index_col=0)['pathologic_stage_label']
    labels = pd.read_csv('../COAD/metadata.csv', index_col=2)['pathologic_stage_label']
    resultDf = pd.DataFrame(columns=['Data', 'Stage', 'Method', 'RFC AUROC', 'GBC AUROC'])
    abundance_micro = pd.read_csv('../COAD/microbiome/microbiome.csv', index_col = 1).iloc[:, 1:]
    
    labels = labels.apply(StageNormalize).dropna()
    abundance_micro = abundance_micro.loc[labels.index]
    clf = ClfSet(abundance_micro.values, labels)
    clf.GBC()
    # Calculating microbiome
    # for data, name in abundanceMicroFiles.items():
    #     print(f'Calculating {data}...')
    #     clfAuc = RunClf(pd.read_csv(f'../COAD/microbiomes/{name}', index_col=0), labelsMicroSer, 'Stage I', 'Stage II')
    #     resultDf.append({'Data': data, 'Stage': 'Stage I vs Stage II', 'RFC AUROC': clfAuc[0], 'GBC AUROC': clfAuc[1]}, ignore_index=True)
    #     clfAuc = RunClf(pd.read_csv(f'../COAD/microbiomes/{name}', index_col=0), labelsMicroSer, 'Stage I', 'Stage III')
    #     resultDf.append({'Data': data, 'Stage': 'Stage I vs Stage III', 'RFC AUROC': clfAuc[0], 'GBC AUROC': clfAuc[1]}, ignore_index=True)
    #     clfAuc = RunClf(pd.read_csv(f'../COAD/microbiomes/{name}', index_col=0), labelsMicroSer, 'Stage I', 'Stage IV')
    #     resultDf.append({'Data': data, 'Stage': 'Stage I vs Stage IV', 'RFC AUROC': clfAuc[0], 'GBC AUROC': clfAuc[1]}, ignore_index=True)
    #     clfAuc = RunClf(pd.read_csv(f'../COAD/microbiomes/{name}', index_col=0), labelsMicroSer, 'Stage II', 'Stage III')
    #     resultDf.append({'Data': data, 'Stage': 'Stage II vs Stage III', 'RFC AUROC': clfAuc[0], 'GBC AUROC': clfAuc[1]}, ignore_index=True)
    #     clfAuc = RunClf(pd.read_csv(f'../COAD/microbiomes/{name}', index_col=0), labelsMicroSer, 'Stage II', 'Stage IV')
    #     resultDf.append({'Data': data, 'Stage': 'Stage II vs Stage IV', 'RFC AUROC': clfAuc[0], 'GBC AUROC': clfAuc[1]}, ignore_index=True)
    #     clfAuc = RunClf(pd.read_csv(f'../COAD/microbiomes/{name}', index_col=0), labelsMicroSer, 'Stage III', 'Stage IV')
    #     resultDf.append({'Data': data, 'Stage': 'Stage III vs Stage IV', 'RFC AUROC': clfAuc[0], 'GBC AUROC': clfAuc[1]}, ignore_index=True)
    #     print('Done')
    # Caculating transcriptome
    # print(f'Calculating transcriptome...')
    # clfAuc = RunClf(abundanceTransFiles, labelsTransSer, 'Stage I', 'Stage II')
    # resultDf = resultDf.append({'Data': 'transcriptome', 'Stage': 'Stage I vs Stage II', 'RFC AUROC': clfAuc[0], 'GBC AUROC': clfAuc[1]}, ignore_index=True)
    # clfAuc = RunClf(abundanceTransFiles, labelsTransSer, 'Stage I', 'Stage III')
    # resultDf = resultDf.append({'Data': 'transcriptome', 'Stage': 'Stage I vs Stage III', 'RFC AUROC': clfAuc[0], 'GBC AUROC': clfAuc[1]}, ignore_index=True)
    # clfAuc = RunClf(abundanceTransFiles, labelsTransSer, 'Stage I', 'Stage IV')
    # resultDf = resultDf.append({'Data': 'transcriptome', 'Stage': 'Stage I vs Stage IV', 'RFC AUROC': clfAuc[0], 'GBC AUROC': clfAuc[1]}, ignore_index=True)
    # clfAuc = RunClf(abundanceTransFiles, labelsTransSer, 'Stage II', 'Stage III')
    # resultDf = resultDf.append({'Data': 'transcriptome', 'Stage': 'Stage II vs Stage III', 'RFC AUROC': clfAuc[0], 'GBC AUROC': clfAuc[1]}, ignore_index=True)
    # clfAuc = RunClf(abundanceTransFiles, labelsTransSer, 'Stage II', 'Stage IV')
    # resultDf = resultDf.append({'Data': 'transcriptome', 'Stage': 'Stage II vs Stage IV', 'RFC AUROC': clfAuc[0], 'GBC AUROC': clfAuc[1]}, ignore_index=True)
    # clfAuc = RunClf(abundanceTransFiles, labelsTransSer, 'Stage III', 'Stage IV')
    # resultDf = resultDf.append({'Data': 'transcriptome', 'Stage': 'Stage III vs Stage IV', 'RFC AUROC': clfAuc[0], 'GBC AUROC': clfAuc[1]}, ignore_index=True)
    # print('Done')
    # resultDf.to_csv('COAD_classfication_result.csv', index=0)