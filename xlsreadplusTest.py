import matlab.engine

eng = matlab.engine.start_matlab()
output = eng.xlsreadplus('yli_data.xlsx',{'13C_g','13C_l'},nargout=4)

#print ("org_avg: average labeling patterns")
#print (output[0])

#print ("covar: covariance matrix")
#print (output[1])

print ("org_raw: raw labeling patterns")
print (output[2])

#print ("avgvar: average variance")
#print (output[3])
eng.quit()