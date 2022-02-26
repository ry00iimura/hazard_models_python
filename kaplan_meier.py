from lifelines import KaplanMeierFitter,CoxPHFitter,AalenAdditiveFitter
import lifelines
from lifelines.statistics import logrank_test
import numpy as np

class CoxHazard:

    def __init__(self,dataset,duration_col,event_col):
        'initialize the dataset'
        self.dataset = dataset
        self.duration = duration_col
        self.event = event_col

    # Kaplan_Meier
    def kaplan(self,label_name = 'KM_estimate'): 
        '''
        Kaplan_Meierはある2個体群の生存時間を比較するときなどに用いられる。X軸が生存時間で、Y軸が累積生存確率。
        累積生存確率は、ある時点において生き残っているサンプルの割合
        よく用いられる場面。薬の投薬による生存時間の比較、故障率、サブスクリプション型の会員のLTV分析
        event は死亡か生存かが確認されたかどうかという値をBool値。途中で観測できなくなった、対象期間内に死亡していない場合は0,死亡は1
        (e.g)  kaplan("duration","event","label_name")
        '''
        kmf = KaplanMeierFitter()
        kmf.fit(
            self.dataset[self.duration]
            ,event_observed=self.dataset[self.event] # event_col <- boolean 
            ,timeline=np.linspace(0,200,200)
            ,label=label_name
            )
        print(kmf.plot())
    
    def logrankTest(self,conditionA,conditionB):
        '''
        ログランク検定
        Kaplan_Meierで生存時間を図示し、2個体群に差がありそうな場合は、その有意差をログランク検定で検定することができる。
        Censoredデータが含まれているため、通常のt検定などが使えない
        ログランク検定をする場合の注意点:2個体群に他の共変量の偏りがある場合は意味がない。例えば、2個体群を薬が投薬されたかどうかで分け、その薬の効果によって生存時間が変わるか検定する。この時、実は、A群だけは全員リハビリを行っていたなどあると意味がないということ
        https://lifelines.readthedocs.io/en/latest/lifelines.statistics.html

        '''
        group_A = self.dataset.query(conditionA)
        group_B = self.dataset.query(conditionB)

        results = logrank_test(
            group_A[self.duration], group_B[self.duration], 
            group_A[self.event], group_B[self.event]
            )

        results.print_summary()

        print("p_value   {}".format(results.p_value))
        print('test_statistic   {}'.format(results.test_statistic))

        return results

    def cph_model(self): 
        '''
        coefが正の値＝危険度が高い。負の値＝その後が良好  ex)cph_graph(modelA,'metastized','event')
        '''
        cph = CoxPHFitter()
        cph.fit(self.dataset, duration_col=self.duration, event_col=self.event)
        print(cph.print_summary())
        cph.plot()

    def aaf_model(self):
        '''
        Aalen Additive Fitter
        '''    
        aaf = AalenAdditiveFitter(fit_intercept=False)
        aaf.fit(self.dataset, duration_col = self.duration, event_col=self.event)
        aaf.plot()

# for debugging
# from lifelines.datasets import load_dd
# data = load_dd()
# data = data[['cowcode2','politycode','duration','observed']]
# data = data.dropna()
# data.head()

if __name__ == '__main__':
    from lifelines import KaplanMeierFitter,CoxPHFitter,AalenAdditiveFitter
    import lifelines
    from lifelines.statistics import logrank_test
    import numpy as np
