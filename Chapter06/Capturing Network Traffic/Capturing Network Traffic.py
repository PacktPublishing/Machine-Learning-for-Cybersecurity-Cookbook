import pyshark
captureTime = 20

print(fileName)
cap = pyshark.LiveCapture(output_file="out.pcap")

cap.sniff(timeout=captureTime)

print(cap)