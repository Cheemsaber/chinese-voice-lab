import sounddevice as sd

print(sd.query_devices())
print("Default input:", sd.query_devices(kind="input"))
print("Host APIs:", sd.query_hostapis())

for index in (1, 4, 9, 18):
    print(index, sd.query_devices(index), "\n")