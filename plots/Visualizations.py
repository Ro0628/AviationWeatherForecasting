#violin plot to visualize weather of time

fig = plt.figure(figsize=(18,6))
LOOKBACK_YEARS = 3
REGISTRATION_YEAR = 2017
filtered_years = car_data[car_data['firstRegistrationYear'] > REGISTRATION_YEAR - LOOKBACK_YEARS]
ax1 = sns.violinplot('firstRegistrationYear', "price", data=filtered_years, hue='modelLine')
ax1.minorticks_on()
ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
ax1.grid(which='minor', axis='x', linewidth=1)