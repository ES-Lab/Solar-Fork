from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

# --------------------------------------------------------
# 4. Evaluation and plot
# --------------------------------------------------------

def evaluate_and_plot (model, dataset):
    print(f" Evaluating {model.name} model...")
    y_pred = model.predict(dataset.X_test)
    y_pred_orig = dataset.scaler_y.inverse_transform(y_pred)
    y_test_orig = dataset.scaler_y.inverse_transform(dataset.y_test)
    
    mse = mean_squared_error(y_test_orig, y_pred_orig)
    mae = mean_absolute_error(y_test_orig, y_pred_orig)

    nRMSE = np.sqrt(mse)/(y_test_orig.max()-y_test_orig.min())
    nMAE = (mae)/(y_test_orig.max()-y_test_orig.min())

    print(f"{model.name} Test MSE: {mse:.4f}")
    print(f"{model.name} Test MAE: {mae:.4f}")
    print(f"{model.name} Test nRMSE: {nRMSE:.4f}")
    print(f"{model.name} Test nMAE: {nMAE:.4f}")


    df_result = pd.DataFrame(
        np.hstack((y_test_orig, y_pred_orig)),
        columns=['Measurement', 'Prediction'])
    
    df_result_error = pd.DataFrame({
        'MSE': mse,
        'MAE': mae,
        'nRMSE':nRMSE,
        'nMAE':nMAE}, index=['error'])

    df_result_error.to_csv(f'{model.name} Forecast Result-Errors.csv')
    df_result.to_csv(f'{model.name} Forecast Result.csv')



    #prepare for plotting
    timestamps_test = df.index[seq_length + split_index:]
    ts=timestamps_test.to_numpy()
    y_test_orig=y_test_orig.reshape(-1)
    a=30*56 #start_index
    b=a+7*56 # end_index, every day consists of 56 instances
    tick_show= 8 # every 15 minutes * ticks_show for lables on Xtick, e.g. 4 means 4*15 min =60 mins 1 hour
    ts_trunc=ts[a:b]
    #y_pred_best_orig_trunc=y_pred_best_orig[a:b] #best
    y_pred_orig_trunc=y_pred_orig[a:b]
    y_test_orig_trunc=y_test_orig[a:b]
    
    #plot
    plt.figure(figsize=(18, 6))
    plt.plot(ts_trunc, y_test_orig_trunc, label='Measurement')
    plt.plot(ts_trunc, y_pred_orig_trunc, label=f'Prediction by {name}')
    plt.legend()
    plt.title(f'Forecast by {name}')
    plt.xlabel('Time')
    plt.ylabel('PV Power [W]')
    plt.xticks(ticks=range(0, len(ts_trunc), tick_show), labels=[ts_trunc[i] for i in range(0, len(ts_trunc), tick_show)],rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{name} Forecast.png')
    plt.show()
    return nRMSE, nMAE 