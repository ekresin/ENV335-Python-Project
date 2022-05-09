#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENV335 Python Project

Author: Emily Kresin
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices
import plotly.graph_objects as go

import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM, PoissonBayesMixedGLM

### Constants ###
kueppers_vals = {'LSA': {'Intercept': [-7.97, 0.59],
                         'Heat': [-2.19, 0.31],
                         'Water': [0.64, 0.30],
                         'Prov': [0.44, 0.18],
                         'Heat x Water': [0.49, 0.30],
                         'Heat x Prov': [-0.16, 0.18],
                         'Water x Prov': [-0.09, 0.16]},
                 'USA': {'Intercept': [-4.34, 0.61],
                         'Heat': [-0.35, 0.10],
                         'Water': [0.34, 0.10],
                         'Prov': [0.15, 0.06],
                         'Heat x Water': [0.02, 0.10],
                         'Heat x Prov': [0.06, 0.06],
                         'Water x Prov': [0.01, 0.06]},
                 'ALP': {'Intercept': [-4.81, 0.92],
                         'Heat': [-0.21, 0.10],
                         'Water': [0.35, 0.10],
                         'Prov': [0.34, 0.06],
                         'Heat x Water': [0.02, 0.10],
                         'Heat x Prov': [-0.02, 0.06],
                         'Water x Prov': [0.01, 0.06]}}

def plot_res(pois_res=None, pois_index=None, bi_res=None, bi_index=None, 
             plot=None, val=None, fname=None):
    fig = go.Figure(layout_title_text=f"{val} coeff. comparison for plot {plot}")
    fig.add_trace(go.Box(x = [0], q1=[pois_res.params[pois_index] - pois_res.fe_sd[pois_index]],
                         q3=[pois_res.params[pois_index] + pois_res.fe_sd[pois_index]],
                         median=[pois_res.params[pois_index]],
                         lowerfence=[pois_res.params[pois_index] - 2*pois_res.fe_sd[pois_index]],
                         upperfence=[pois_res.params[pois_index] + 2*pois_res.fe_sd[pois_index]],
                         boxpoints=False,
                         name="SM Poisson"))
    fig.add_trace(go.Box(x = [1], q1=[bi_res.params[bi_index] - bi_res.fe_sd[bi_index]],
                         q3=[bi_res.params[bi_index] + bi_res.fe_sd[bi_index]],
                         median=[bi_res.params[bi_index]],
                         lowerfence=[bi_res.params[bi_index] - 2*bi_res.fe_sd[bi_index]],
                         upperfence=[bi_res.params[bi_index] + 2*bi_res.fe_sd[bi_index]],
                         boxpoints=False,
                         name="SM Binomial"))
    fig.add_trace(go.Box(x = [2], q1=[kueppers_vals[plot][val][0] - kueppers_vals[plot][val][1]],
                         q3=[kueppers_vals[plot][val][0] + kueppers_vals[plot][val][1]],
                         median=[kueppers_vals[plot][val][0]],
                         lowerfence=[kueppers_vals[plot][val][0] - 2*kueppers_vals[plot][val][1]],
                         upperfence=[kueppers_vals[plot][val][0] + 2*kueppers_vals[plot][val][1]],
                         boxpoints=False,
                         name="Kueppers et al."))
    fig.write_html(fname)

if __name__ == "__main__": 

#=======================================================================
 """Table 2 in Publication & S1 in Supporting"""
 # =============================================================================
 """LSA Engelmann Poisson"""
published_seedling_df = pd.read_excel('./paper_seedling_data_2010_to_2014.xlsx')
lsa_df = published_seedling_df.loc[published_seedling_df['Plot'] <=20]
lsa_pien_df = lsa_df.loc[lsa_df['Species'] == 'PiEn'] 
y, X = dmatrices('sS1 ~ Heat + Water + Provenance + Heat*Water + Heat*Provenance + Water*Provenance', 
                  data=lsa_pien_df, return_type='dataframe')

y_temp, exog_dv = dmatrices('sS1 ~ C(Cohort)/C(Plot) - 1', data=lsa_pien_df, return_type='dataframe')

ident = np.arange(1, exog_dv.shape[1]+1) 
mod = PoissonBayesMixedGLM(y, X, exog_dv, ident, vcp_p=.5)

res = mod.fit_vb()
#print(res.summary())


"""LSA Engelmann Binomial"""
published_seedling_df = pd.read_excel('./paper_seedling_data_2010_to_2014.xlsx')
lsa_df = published_seedling_df.loc[published_seedling_df['Plot'] <= 20]
lsa_pien_df = lsa_df.loc[lsa_df['Species'] == 'PiEn'] 
lsa_pien_df_trim = lsa_pien_df[["Heat","Water","Plot","Cohort","Provenance","SeedsSown",
                                "kills", "sS1"]]
lsa_pien_df_bin = pd.DataFrame(columns=["Heat", "Water", "Plot", 
                                        "Cohort", "Provenance", "Survived"])

rows_added = 0
for i, row in lsa_pien_df_trim.iterrows():
    try:                                        # sometimes there is a NA value in the
        num_sown = int(row["SeedsSown"])        # spreadsheet for these values.
    except:                                     # these try/except statments try to
        num_sown = 0                            # convert the values in these columns
    try:                                        # to integers and give instructions
        num_killed = int(row["kills"])          # on what to do if the value cannot
    except ValueError:                          # be converted (i.e. when the value
        num_killed = 0                          # is NA)
    num_snk = num_sown - num_killed
    try:
        num_surv = int(row["sS1"])
    except ValueError:
        continue
    num_not_surv = num_snk - num_surv
    surv_df = pd.DataFrame([[row["Heat"],
                             row["Water"],
                             row["Plot"],
                             row["Cohort"],
                             row["Provenance"],
                             1]]*num_surv, 
                             index=range(rows_added, rows_added + num_surv), 
                             columns=lsa_pien_df_bin.columns)
    rows_added += num_surv
    not_surv_df = pd.DataFrame([[row["Heat"],
                                 row["Water"],
                                 row["Plot"],
                                 row["Cohort"],
                                 row["Provenance"],
                                 0]]*num_not_surv, 
                                 index=range(rows_added, rows_added + num_not_surv), 
                                 columns=lsa_pien_df_bin.columns)
    rows_added += num_not_surv
    lsa_pien_df_bin = lsa_pien_df_bin.append(surv_df)
    lsa_pien_df_bin = lsa_pien_df_bin.append(not_surv_df)
y, X = dmatrices('Survived ~ Heat + Water + Provenance + Heat*Water + Heat*Provenance + Water*Provenance', 
                data=lsa_pien_df_bin, return_type='dataframe')
y = y["Survived[1]"]
y_temp, exog_dv = dmatrices('Survived ~ C(Cohort)/C(Plot) - 1', data=lsa_pien_df_bin, return_type='dataframe')
ident = np.arange(1, exog_dv.shape[1]+1) # create the model using the Poisson-Bayes General Mixed Effects Linear Model
mod = BinomialBayesMixedGLM(y, X, exog_dv, ident, vcp_p=.5)
bi_res = mod.fit_map()
   
plot_res(pois_res=res, pois_index=0, bi_res=bi_res, bi_index=0, plot='LSA', val='Intercept', fname='lsa_pien_intercept.html') 
plot_res(pois_res=res, pois_index=2, bi_res=bi_res, bi_index=1, plot='LSA', val='Heat', fname='lsa_pien_heat.html')
plot_res(pois_res=res, pois_index=4, bi_res=bi_res, bi_index=2, plot='LSA', val='Water', fname='lsa_pien_water.html')
plot_res(pois_res=res, pois_index=1, bi_res=bi_res, bi_index=3, plot='LSA', val='Prov', fname='lsa_pien_prov.html')
plot_res(pois_res=res, pois_index=6, bi_res=bi_res, bi_index=4, plot='LSA', val='Heat x Water', fname='lsa_pien_heat_water.html')
plot_res(pois_res=res, pois_index=3, bi_res=bi_res, bi_index=5, plot='LSA', val='Heat x Prov', fname='lsa_pien_heat_prov.html')
plot_res(pois_res=res, pois_index=5, bi_res=bi_res, bi_index=6, plot='LSA', val='Water x Prov', fname='lsa_pien_water_prov.html')

# =============================================================================
"""USA Engelmann Poisson"""
usa_df = published_seedling_df.loc[(published_seedling_df['Plot'] >=21)*(published_seedling_df['Plot'] <=40)] 
usa_pien_df = usa_df.loc[usa_df['Species'] == 'PiEn']
y, X = dmatrices('sS1 ~ Heat + Water + Provenance + Heat*Water + Heat*Provenance + Water*Provenance', 
                data=usa_pien_df, return_type='dataframe')
y_temp, exog_dv = dmatrices('sS1 ~ C(Cohort)/C(Plot) - 1', data=usa_pien_df, return_type='dataframe')
ident = np.arange(1, exog_dv.shape[1]+1)
mod = PoissonBayesMixedGLM(y, X, exog_dv, ident, vcp_p=.5)
res = mod.fit_vb()
print(res.summary())


"""USA Engelmann Binomial"""
published_seedling_df = pd.read_excel('./paper_seedling_data_2010_to_2014.xlsx')
usa_df = published_seedling_df.loc[(published_seedling_df['Plot'] >=21)*(published_seedling_df['Plot'] <=40)]
usa_pien_df = usa_df.loc[usa_df['Species'] == 'PiEn'] 
usa_pien_df_trim = usa_pien_df[["Heat","Water","Plot","Cohort","Provenance","SeedsSown",
                            "kills", "sS1"]]
usa_pien_df_bin = pd.DataFrame(columns=["Heat", "Water", "Plot", 
                                    "Cohort", "Provenance", "Survived"]) 
rows_added = 0
for i, row in usa_pien_df_trim.iterrows():
    try:                                        
        num_sown = int(row["SeedsSown"])        
    except:                                     
        num_sown = 0                            
    try:                                        
        num_killed = int(row["kills"])          
    except ValueError:                          
        num_killed = 0                          
        num_snk = num_sown - num_killed
    try:
        num_surv = int(row["sS1"])
    except ValueError:
        continue
    num_not_surv = num_snk - num_surv
    surv_df = pd.DataFrame([[row["Heat"],
                             row["Water"],
                             row["Plot"],
                             row["Cohort"],
                             row["Provenance"],
                             1]]*num_surv, 
                             index=range(rows_added, rows_added + num_surv), 
                             columns=usa_pien_df_bin.columns)
    rows_added += num_surv
    not_surv_df = pd.DataFrame([[row["Heat"],
                                 row["Water"],
                                 row["Plot"],
                                 row["Cohort"],
                                 row["Provenance"],
                                 0]]*num_not_surv, 
                                 index=range(rows_added, rows_added + num_not_surv), 
                                 columns=usa_pien_df_bin.columns)
    rows_added += num_not_surv
    usa_pien_df_bin = usa_pien_df_bin.append(surv_df)
    usa_pien_df_bin = usa_pien_df_bin.append(not_surv_df)
y, X = dmatrices('Survived ~ Heat + Water + Provenance + Heat*Water + Heat*Provenance + Water*Provenance', 
            data=usa_pien_df_bin, return_type='dataframe')
y = y["Survived[1]"]
y_temp, exog_dv = dmatrices('Survived ~ C(Cohort)/C(Plot) - 1', data=usa_pien_df_bin, return_type='dataframe')
mod = BinomialBayesMixedGLM(y, X, exog_dv, ident, vcp_p=.5)
bi_res = mod.fit_map()
print(bi_res.summary())

plot_res(pois_res=res, pois_index=0, bi_res=bi_res, bi_index=0, plot='USA', val='Intercept', fname='usa_pien_intercept.html') 
plot_res(pois_res=res, pois_index=2, bi_res=bi_res, bi_index=1, plot='USA', val='Heat', fname='usa_pien_heat.html')
plot_res(pois_res=res, pois_index=4, bi_res=bi_res, bi_index=2, plot='USA', val='Water', fname='usa_pien_water.html')
plot_res(pois_res=res, pois_index=1, bi_res=bi_res, bi_index=3, plot='USA', val='Prov', fname='usa_pien_prov.html')
plot_res(pois_res=res, pois_index=6, bi_res=bi_res, bi_index=4, plot='USA', val='Heat x Water', fname='usa_pien_heat_water.html')
plot_res(pois_res=res, pois_index=3, bi_res=bi_res, bi_index=5, plot='USA', val='Heat x Prov', fname='usa_pien_heat_prov.html')
plot_res(pois_res=res, pois_index=5, bi_res=bi_res, bi_index=6, plot='USA', val='Water x Prov', fname='usa_pien_water_prov.html')

# =============================================================================
"""ALP Engelmann Poisson"""
alp_df =published_seedling_df.loc[published_seedling_df['Plot'] >=41]
alp_pien_df = alp_df.loc[alp_df['Species'] == 'PiEn']
y, X = dmatrices('sS1 ~ Heat + Water + Provenance + Heat*Water + Heat*Provenance + Water*Provenance', 
                data=alp_pien_df, return_type='dataframe')
y_temp, exog_dv = dmatrices('sS1 ~ C(Cohort)/C(Plot) - 1', data=alp_pien_df, return_type='dataframe')
ident = np.arange(1, exog_dv.shape[1]+1)
mod = PoissonBayesMixedGLM(y, X, exog_dv, ident, vcp_p=.5)
res = mod.fit_vb()
print (res.summary())


"""ALP Engelmann Binomial"""
alp_pien_df_trim = alp_pien_df[["Heat","Water","Plot","Cohort","Provenance","SeedsSown",
                                "kills", "sS1"]]
alp_pien_df_bin = pd.DataFrame(columns=["Heat", "Water", "Plot", 
                                        "Cohort", "Provenance", "Survived"]) 
rows_added = 0
for i, row in alp_pien_df_trim.iterrows():
    try:                                        
        num_sown = int(row["SeedsSown"])        
    except:                                     
        num_sown = 0                            
    try:                                        
        num_killed = int(row["kills"])          
    except ValueError:                          
        num_killed = 0                          
        num_snk = num_sown - num_killed
    try:
        num_surv = int(row["sS1"])
    except ValueError:
        continue
    num_not_surv = num_snk - num_surv
    surv_df = pd.DataFrame([[row["Heat"],
                             row["Water"],
                             row["Plot"],
                             row["Cohort"],
                             row["Provenance"],
                             1]]*num_surv, 
                             index=range(rows_added, rows_added + num_surv), 
                             columns=alp_pien_df_bin.columns)
    rows_added += num_surv
    not_surv_df = pd.DataFrame([[row["Heat"],
                                 row["Water"],
                                 row["Plot"],
                                 row["Cohort"],
                                 row["Provenance"],
                                 0]]*num_not_surv, 
                                 index=range(rows_added, rows_added + num_not_surv), 
                                 columns=alp_pien_df_bin.columns)
    rows_added += num_not_surv
    alp_pien_df_bin = alp_pien_df_bin.append(surv_df)
    alp_pien_df_bin = alp_pien_df_bin.append(not_surv_df)
    
y, X = dmatrices('Survived ~ Heat + Water + Provenance + Heat*Water + Heat*Provenance + Water*Provenance', 
                data=alp_pien_df_bin, return_type='dataframe')
y = y["Survived[1]"]
y_temp, exog_dv = dmatrices('Survived ~ C(Cohort)/C(Plot) - 1', data=alp_pien_df_bin, return_type='dataframe')
mod = BinomialBayesMixedGLM(y, X, exog_dv, ident, vcp_p=.5)
bi_res = mod.fit_map()
print(bi_res.summary())

plot_res(pois_res=res, pois_index=0, bi_res=bi_res, bi_index=0, plot='ALP', val='Intercept', fname='alp_pien_intercept.html') 
plot_res(pois_res=res, pois_index=2, bi_res=bi_res, bi_index=1, plot='ALP', val='Heat', fname='alp_pien_heat.html')
plot_res(pois_res=res, pois_index=4, bi_res=bi_res, bi_index=2, plot='ALP', val='Water', fname='alp_pien_water.html')
plot_res(pois_res=res, pois_index=1, bi_res=bi_res, bi_index=3, plot='ALP', val='Prov', fname='alp_pien_prov.html')
plot_res(pois_res=res, pois_index=6, bi_res=bi_res, bi_index=4, plot='ALP', val='Heat x Water', fname='alp_pien_heat_water.html')
plot_res(pois_res=res, pois_index=3, bi_res=bi_res, bi_index=5, plot='ALP', val='Heat x Prov', fname='alp_pien_heat_prov.html')
plot_res(pois_res=res, pois_index=5, bi_res=bi_res, bi_index=6, plot='ALP', val='Water x Prov', fname='alp_pien_water_prov.html')




