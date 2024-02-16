import json

filename1 = "results/IOT_DNL/mpi/centralized_async/4_200*/server.json"
filename2 = "results/IOT_DNL/zenoh/centralized_async/4_200/server.json"
model_1 = json.load(open(filename1))
model_2 = json.load(open(filename2))


for i in range(len(model_1["mcc"])):
    if model_1["mcc"][i] != model_2["mcc"][i]:
        print(i, model_1["mcc"][i], model_2["mcc"][i],  model_1["mcc"][i]-model_2["mcc"][i])