import json

filename1 = "results/Slicing5G/mpi/decentralized_sync/8_300_1_256/server.json"
filename2 = "results/Slicing5G/zenoh/decentralized_sync/8_300_1_256/server.json"
model_1 = json.load(open(filename1))
model_2 = json.load(open(filename2))


for i in range(len(model_1["mcc"])):
    if model_1["mcc"][i] != model_2["mcc"][i]:
        print(i, model_1["mcc"][i], model_2["mcc"][i],  model_1["mcc"][i]-model_2["mcc"][i])