n = 8; 
m = 10000;
upper_bound = 4;
for i= 1:m
    i
    tspdatacsv(n,['tspdata',num2str(i),'_upper_bound',num2str(upper_bound),'_cities_',num2str(n),'.dat'],upper_bound);
end
