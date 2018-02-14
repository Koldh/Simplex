function tspdatacsv(n,name,upper_bound)
var = n*n-n;
tvar = n*n - 1;
c = unidrnd(upper_bound,1,var);
temp = zeros(1,n-1);
c = [c,temp];
Aeq = zeros(2*n,tvar);
for i=1:n
    for j = 1:n
        if i~=j
            index = indexcalc(i,j,n);
            Aeq(i,index) = 1;
        end
    end
end
for j=1:n
    for i = 1:n
        if i~=j
            index = indexcalc(j,i,n);
            Aeq(n+i,index) = 1;
        end
    end
end
beq = ones(1,2*n);
iteration = 0;
Ain = zeros((n-1)*(n-2),tvar);
for i = 2:n
    for j = 2:n
        if i~=j
            iteration = iteration + 1;
            index = var + i - 1;
            Ain(iteration,index) = 1;
            index = var + j - 1;
            Ain(iteration,index) = -1;
            index = indexcalc(i,j,n);
            Ain(iteration,index) = n;
        end
    end
end
bin = zeros(1,(n-1)*(n-2))+(n-1);

rowsize = 1 + 2*n + 1 + (n-1)*(n-2) + 1;
columnsize = max(tvar,2*n);
data = cell(rowsize,columnsize);
rowindex = 1;
for i=1:tvar
    data(rowindex,i) = {c(i)};
end
for i=1:2*n
    for j = 1:tvar
        data(rowindex+i,j) = {Aeq(i,j)};
    end
end
rowindex = rowindex + 2*n;
for i=1:2*n
    data(rowindex+1,i) = {beq(i)};
end
rowindex = rowindex + 1;
for i=1:(n-1)*(n-2)
    for j = 1:tvar
        data(rowindex+i,j) = {Ain(i,j)};
    end
end
rowindex = rowindex + (n-1)*(n-2);
for i=1:(n-1)*(n-2)
    data(rowindex+1,i) = {bin(i)};
end
csvwrite(name,data);
