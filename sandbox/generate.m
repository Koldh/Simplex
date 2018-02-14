n =10; 
m =50;
upper_bound = 4
for i= 1:m
    tspdatacsv(n,['tspdata',num2str(i),'_upper_bound',num2str(upper_bound),'_cities_',num2str(n)],upper_bound)
end
