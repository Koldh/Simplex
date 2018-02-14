function index = indexcalc(i,j,n)
index = (i-1)*(n-1);
if j<i
    index = index + j;
elseif j > i
    index = index + j -1;
else
    display(10000)
end
end
