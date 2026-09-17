# 1. Historical Data Registry (Word-Based for Naive Bayes)
history_data = [
    ["High",   "Dropping", "Cool", "Yes"],
    ["High",   "Stable",   "Warm", "No"],
    ["Normal", "Dropping", "Cool", "Yes"],
    ["High",   "Dropping", "Warm", "Yes"],
    ["Normal", "Stable",   "Warm", "No"],
    ["Normal", "Stable",   "Cool", "No"],
    ["High",   "Stable",   "Cool", "No"],
    ["Normal", "Dropping", "Warm", "Yes"]
]

# 2. Conversion Rules: Turn raw numerical sensor readings into category words
def convert_humidity_to_word(numerical_humidity):
    # If humidity is 70% or higher, classify it as High
    if numerical_humidity >= 70.0:
        return "High"
    else:
        return "Normal"

def convert_temp_to_word(numerical_celsius):
    # If temperature is below 20 degrees Celsius, classify it as Cool
    if numerical_celsius < 20.0:
        return "Cool"
    else:
        return "Warm"

# 3. LIVE SENSOR INPUTS (Mix of exact numbers and words!)
raw_sensor_humidity = 84.5       # Real number from sensor (%)
current_sensor_pressure = "Dropping" # Direct word status from sensor trend
raw_sensor_temperature = 18.2    # Real number from sensor (Celsius)

# Apply our conversion rules to the numerical variables
current_sensor_humidity = convert_humidity_to_word(raw_sensor_humidity)
current_sensor_temp = convert_temp_to_word(raw_sensor_temperature)


# 4. Math Processing (Naive Bayes Engine)
total_days = len(history_data)
count_rain_yes = 0
count_rain_no = 0

for day in history_data:
    if day[3] == "Yes":
        count_rain_yes += 1
    else:
        count_rain_no += 1

prior_rain = count_rain_yes / total_days
prior_clear = count_rain_no / total_days

match_hum_yes, match_pres_yes, match_temp_yes = 0, 0, 0
match_hum_no,  match_pres_no,  match_temp_no  = 0, 0, 0

for day in history_data:
    if day[3] == "Yes":
        if day[0] == current_sensor_humidity: match_hum_yes += 1
        if day[1] == current_sensor_pressure: match_pres_yes += 1
        if day[2] == current_sensor_temp:     match_temp_yes += 1
    else:
        if day[0] == current_sensor_humidity: match_hum_no += 1
        if day[1] == current_sensor_pressure: match_pres_no += 1
        if day[2] == current_sensor_temp:     match_temp_no += 1

p_hum_yes  = match_hum_yes / count_rain_yes
p_pres_yes = match_pres_yes / count_rain_yes
p_temp_yes = match_temp_yes / count_rain_yes

p_hum_no   = match_hum_no / count_rain_no
p_pres_no  = match_pres_no / count_rain_no
p_temp_no  = match_temp_no / count_rain_no

score_rain  = prior_rain * p_hum_yes * p_pres_yes * p_temp_yes
score_clear = prior_clear * p_hum_no * p_pres_no * p_temp_no

total_score = score_rain + score_clear
prob_rain = (score_rain / total_score) * 100
prob_clear = (score_clear / total_score) * 100


# 5. Display the Results
print("========================================")
print("     HYBRID SENSOR DATA SIMULATOR       ")
print("========================================")
print("Raw Hardware Values Read:")
print(f" -> Humidity Sensor   : {raw_sensor_humidity}%")
print(f" -> Pressure Trend    : '{current_sensor_pressure}'")
print(f" -> Temp Sensor       : {raw_sensor_temperature}°C\n")

print("Converted Words for Naive Bayes Processing:")
print(f" -> Humidity: {current_sensor_humidity} | Pressure: {current_sensor_pressure} | Temp: {current_sensor_temp}\n")

print(f"Calculated Probability of Rain : {prob_rain:.1f}%")
print(f"Calculated Probability of Clear: {prob_clear:.1f}%\n")

print("System Response:")
if prob_rain > prob_clear:
    print(" >>> SYSTEM ACTION: RAIN PREDICTED - Warning LED activated.")
else:
    print(" >>> SYSTEM ACTION: CLEAR SKIES - Indicator turned off.")
print("========================================")
