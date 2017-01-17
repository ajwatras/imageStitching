file = open('test.txt','r');

my_text = file.readlines();
barrel = my_text[3::13];
gray = my_text[4::13];
upload = my_text[5::13];
DandD = my_text[6::13];
Match = my_text[7::13];
calc = my_text[8::13];
applyH = my_text[9::13];
stitch = my_text[10::13];
total = my_text[11::13];
fps = my_text[12::13];

#print barrel.split();

for k in range(0,len(barrel)):
	barrel[k] = barrel[k].split()[3];
	gray[k] = gray[k].split()[3];
	upload[k] = upload[k].split()[4];
	DandD[k] = DandD[k].split()[3];
	Match[k] = Match[k].split()[2];
	calc[k] = calc[k].split()[2];
	applyH[k] = applyH[k].split()[2];
	total[k] = total[k].split()[5];
	fps[k] = fps[k].split()[3];


n = len(barrel);


temp_barrel = 0;
temp_gray = 0;
temp_upload = 0;
temp_DandD = 0;
temp_Match = 0;
temp_calc = 0;
temp_apply = 0;
temp_total = 0;
temp_fps = 0;
for k in range(0,n):
	temp_barrel = temp_barrel + float(barrel[k])/n;
	temp_gray = temp_gray + float(gray[k])/n;
	temp_upload = temp_upload + float(upload[k])/n;
	temp_DandD = temp_DandD + float(DandD[k])/n;
	temp_Match = temp_Match + float(Match[k])/n;
	temp_calc = temp_calc + float(calc[k])/n;
	temp_apply = temp_apply + float(applyH[k])/n;
	temp_total = temp_total + float(total[k])/n;
	temp_fps = temp_fps + float(fps[k])/n;




print "\n Frames: ", n;
print "Removing Barrel distortion: ", temp_barrel, "ms";
print "Converting to Grayscale: ", temp_gray, "ms";
print "Uploading to GPU: ", temp_upload, "ms";
print "Detecting Feature Points: ", temp_DandD, "ms";
print "Matching Feature Points: ", temp_Match, "ms";
print "Calculating Homography: ", temp_calc, "ms";
print "Applying Homography: ",temp_apply,"ms";
print "Total Computation Time: ",temp_total,"ms";
print "Frames Per Second: ",1/temp_total*1000, "ms";
