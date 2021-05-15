#this is meant to add content to fill the intents.json file
#for the AI Chatbot.

import json

def write_json(data, filename="intents.json"):
    with open(filename, "w") as f:
        json.dump(data, f, indent=1)

print("Input things to put into Intents file\n")
tag = input("Enter tag (enter \'q\' when done): ")
patterns = []
responses = []
labels = []

flag = False
while flag == False:
    pattern = input("Enter Patterns (enter \'q\' when done): ")
    if pattern == "q":
        flag = True
        break
    else:
        patterns.append(pattern)

flag = False
while flag == False:
    response = input("Enter Responses (enter \'q\' when done): ")
    if response == "q":
        flag = True
        break
    else:
        responses.append(response)

with open("intents.json") as file:
    data = json.load(file)
    temp = data["intents"]
    for intent in temp:
        if intent["tag"] not in labels:
            labels.append(intent["tag"])
    if tag not in labels:
        y = {"tag": tag, "patterns": patterns, "responses": responses}
        temp.append(y)

write_json(data)