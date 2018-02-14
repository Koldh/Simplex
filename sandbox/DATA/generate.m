n =10; 
m =5;
upper_bound = 4;
for i= 1:m
    tspdatacsv(n,['tspdata',num2str(i),'_upper_bound',num2str(upper_bound),'_cities_',num2str(n),'.dat'],upper_bound);
end
