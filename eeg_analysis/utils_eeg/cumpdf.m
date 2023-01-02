
function [p95]=cumpdf(data_x)
    level=0.95;
    [pdf,bin] = hist(data_x, 1:max(data_x));
    pdf = pdf./sum(pdf);
    pdf_cumsum = cumsum(pdf);
    find_var = find(pdf_cumsum<=level);
    p95 = interp1(pdf_cumsum(find_var(end):(find_var(end)+1)),bin(find_var(end):(find_var(end)+1)),level,'spline');
end