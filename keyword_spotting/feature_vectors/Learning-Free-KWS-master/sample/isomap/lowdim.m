descriptors = matfile('iso_desc','Writable',true);
dim = 30;
options.dims = 1:dim;
[Y, R, E] = Isomap(descriptors.distances,'k',5,options);
info.wDescriptors = zeros(2501,dim);
A = cell2mat(Y.coords(dim,1));
for i = 1:2501
    
    info.wDescriptors(i,:) = A(:,i);
end
descriptors.wDescriptors = info.wDescriptors;