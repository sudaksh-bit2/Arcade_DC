

def import_data_and_create_master():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import matplotlib.dates as mdates
    
    
    ## Read in data and housekeeping
    
    accounts = pd.read_csv('account_dat.csv')
    apps=pd.read_csv('app_dat.csv')
    in_app_cont=pd.read_csv('in-app_dat.csv')
    devices=pd.read_csv('device_ref.csv')
    categories=pd.read_csv('category_ref.csv')
    trans=pd.read_csv('transaction_dat.csv')

    accounts.create_dt = pd.to_datetime(accounts.create_dt)
    trans.create_dt = pd.to_datetime(trans.create_dt)

    # Create year_month to analyze trends across time
    accounts['year_mon_account'] = pd.to_datetime(accounts.create_dt).dt.to_period('M')
    trans['year_mon_trans'] = pd.to_datetime(trans.create_dt).dt.to_period('M')

    #rename vars to more logical names
    accounts=accounts.rename(columns={'create_dt':'act_create_dt'})
    categories=categories.rename(columns={'category_name':'app_category'})
    apps=apps.rename(columns={'content_id':'app_id'})
    in_app_cont=in_app_cont.rename(columns={'parent_app_content_id':'app_id','type':'content_type'})
    trans=trans.rename(columns={'device_id':'purchase_device_id', 'create_dt':'trans_dt'})
    
    ## Map apps to category
    apps_cat=pd.merge(apps,categories,on='category_id').drop(columns=['category_id'])
    ## Map apps to in_app_cont
    apps_wt_cont = pd.merge(apps_cat,in_app_cont,on='app_id',how='left')
    ## Map transactions and app_wt_cont
    trans_dev=pd.merge(trans,devices,left_on='purchase_device_id',right_on='device_id',how='left').drop(columns=['device_id'])
    trans_master=pd.merge(trans_dev,apps_wt_cont.drop(columns=['device_id']),on='content_id',how='left')
    ## Map trans_master with accounts
    master=pd.merge(trans_master,accounts,on='acct_id',how='left')
    master['age_on_pltfm']=(master.trans_dt-master.act_create_dt).dt.days
    
    return master

def generate_features(date_of_segmentation,dataset, all_agg_functions=['mean','count','sum'],all_metrics=['price']):
    import pandas as pd
    import numpy as np
    import datetime
    
    dataset.fillna({'app_name':'unknown',
               'app_id':'unknown',
               'app_category':'unknown',
               'content_type':'unknown',
               'payment_type':'unknown',
               }, inplace=True)
    
    date_of_segmentation=pd.to_datetime(date_of_segmentation)
    master_sub=dataset[(dataset.trans_dt<=date_of_segmentation)]

    feat_master=pd.DataFrame(index=master_sub.acct_id.unique())
    feat_master.head()

    # Index for time window
    #appending 100000 days to get all historical data
    days_prev_all=[7,14,30,60]
    days_prev_all.append(100000)
    print('Days_prev: '+str(days_prev_all))
    app_categories_all=master_sub.app_category.unique().tolist()
    app_categories_all.append('All')
    print('App_categories: '+ str(app_categories_all))
    devices_all = master_sub.device_name.unique().tolist()
    devices_all.append('All')
    print('Devices: '+ str(devices_all))
    content_all = master_sub.content_type.unique().tolist()
    content_all.append('All')
    print('Content types: '+ str(content_all))



    for days_prev in days_prev_all:
        for app_cat in app_categories_all:
            for device in devices_all:
                for content_type in content_all:
                    #Index for days_prev
                    # in case of 'All' generate a series with all True so that no filtering is done
                    time_window=datetime.timedelta(days_prev)
                    time_filter_index=np.where(days_prev=='All',~(master_sub.trans_dt.isnull()),(master_sub.trans_dt>(date_of_segmentation-time_window)))

                    # Index for App Category
                    # in case of 'All' generate a series with all True so that no filtering is done
                    app_filter_index = np.where(app_cat=='All',~(master_sub.app_category.isnull()),master_sub.app_category==app_cat)

                    # Index for Device
                    # in case of 'All' generate a series with all True so that no filtering is done
                    device_filter_index = np.where(device=='All',~(master_sub.device_name.isnull()), master_sub.device_name==device)

                    # Index for Content Type
                    # in case of 'All' generate a series with all True so that no filtering is done
                    content_filter_index =  np.where(content_type=='All',~(master_sub.content_type.isnull()), master_sub.content_type==content_type)

                    #All filters
                    all_filters=device_filter_index & app_filter_index & time_filter_index & content_filter_index

                    # String to use in feature name
                    cuts_string='_days_'+str(days_prev)+'_app_cat_'+app_cat+'_device_'+device+'_cont_type_'+content_type

                    # FILTER THE DATA
                    master_filt=master_sub[all_filters]

                    for agg_function in all_agg_functions:
                        for metric in all_metrics:
                            # Check if after filtering there are no transactions left
                            if (all_filters.sum()==0):
                                feat_name=str(metric+'_'+agg_function+cuts_string)
                                print('NO TRANSACTIONS IN THIS CUT: '+ feat_name)
                            else:
                                # AGGREGATION
                                feat_name=str(metric+'_'+agg_function+cuts_string)
                                print('Feature: '+feat_name+' created')
                                feat_df=master_filt.groupby('acct_id').agg({metric:agg_function}).rename(columns={metric:feat_name})
                                feat_master=feat_master.join(feat_df)
                                print('Feature Number: '+str(feat_master.shape[1]))
    
    print(str(feat_master.shape[1])+' features created')
    
    feat_master.reset_index().rename(columns={'index':'acct_id'}).to_csv('feat_master.csv')
    feat_master = feat_master.reset_index().rename(columns={'index':'acct_id'})
    
    return feat_master
    

    
def scaled_PCA_kmeans(data,nclus):
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    master_numpy = data.fillna(0).drop('acct_id',axis=1).to_numpy()
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    
    master_numpy=data.fillna(0).drop('acct_id',axis=1).to_numpy()
    
    scaler = StandardScaler()
    scaled_data=scaler.fit_transform(master_numpy)

    pca = PCA(n_components=0.9)
    pca.fit(scaled_data)
    pca_data=pca.transform(scaled_data)

    #Elbow curve
    inertias=[]
    for k in range(1,20):
        model = KMeans(n_clusters=k)

        # Fit model to samples
        model.fit(pca_data)

        # Append the inertia to the list of inertias
        inertias.append(model.inertia_)

    plt.plot(range(1,20), inertias, '-p')
    plt.xlabel('number of clusters, k')
    plt.ylabel('inertia')
    plt.show()
    
    kmeans = KMeans(n_clusters=nclus)
    kmeans.fit(pca_data)
    labels = kmeans.predict(pca_data)
    
    pca_df=pd.DataFrame(pca_data)
    
    plt.scatter(pca_df[0], pca_df[1], c=labels)
    plt.title('The acct_id is segmented into ' +str(nclus)+ ' clusters')
    plt.show()

    return labels