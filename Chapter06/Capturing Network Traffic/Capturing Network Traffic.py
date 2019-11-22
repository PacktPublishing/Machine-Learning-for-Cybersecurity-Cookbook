import pyshark

capture_time = 20

import datetime

start = datetime.datetime.now()
end = start+datetime.timedelta(seconds=capture_time)
file_name = "networkTrafficCatpureFrom"+str(start).replace(" ", "T")+"to"+str(end).replace(" ","T")+".pcap"

print(fileName)
cap = pyshark.LiveCapture(output_file="out.pcap")

cap.sniff(timeout=capture_time)

print(cap)