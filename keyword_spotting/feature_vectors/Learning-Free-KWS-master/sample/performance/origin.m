load('SampleRelevantList','queriesInfo');
wmatname = 'words_POG_Sequential';
qmatname = 'queries_POG_Sequential_0';
queriesmat = matfile(['temp_descriptors/' qmatname]);
wordsmat = matfile(['temp_descriptors/' wmatname]);

qnames = queriesmat.names;
N = numel(qnames);
wnames = wordsmat.names;

percRef = 0.1;

% Method Variant
variant = 0;
disp('variant value')
disp(variant)
ns = 1;

wrefaspect = wordsmat.refaspect;

% PCA
Aref = wordsmat.Aref;
if ~variant
    A = wordsmat.A;
end

refNdim = size(Aref,2);
tmpmat = matfile(['temp_descriptors/' 'temp'],'Writable',true);
tmpmat.wRefDescriptors(numel(wnames),refNdim) = 0;
%tmpmat.qRefDescriptors(numel(qnames),size(Aref,2)) = 0;
%tmpmat.wRefDescriptors(1,numel(wnames)) = {[]};
tmpmat.qRefDescriptors(1,numel(qnames)) = {[]};
tmpmat.wDescriptors(1,numel(wnames)) = {[]};
tmpmat.qDescriptors(1,numel(qnames)) = {[]};


batchsize = 200;
% words pca
cnt = 1;
while (cnt <= numel(wnames))
    range = cnt:min(numel(wnames),cnt+batchsize-1);
    cnt = cnt + batchsize;    
    
    %tmpmat.wRefDescriptors(range,:) = wordsmat.refDescriptors(range,:)*Aref;
    %tmpmat.wRefDescriptors(1,range) = cellfun(@(x) x*Aref,wordsmat.refDescriptors(1,range),'uniformoutput',0);
    tmpbatchrefdesc = wordsmat.refDescriptors(1,range);
    tmpmat.wRefDescriptors(range,:) = cat(1,tmpbatchrefdesc{:})*Aref; 
   
    if ~variant
        tmpmat.wDescriptors(1,range) = cellfun(@(x) x*A,wordsmat.descriptors(1,range),'uniformoutput',0);
    end
end

% queries pca
cnt = 1;
while (cnt <= numel(qnames))
    range = cnt:min(numel(qnames),cnt+batchsize-1);
    cnt = cnt + batchsize;    
    tmpmat.qRefDescriptors(1,range) = cellfun(@(x) x*Aref,queriesmat.refDescriptors(1,range),'uniformoutput',0);
    
    if ~variant
        tmpmat.qDescriptors(1,range) = cellfun(@(x) x*A,queriesmat.descriptors(1,range),'uniformoutput',0);
    end
end

wrdesc = tmpmat.wRefDescriptors;
if ~variant
    wdesc = tmpmat.wDescriptors;
end

AP = zeros(1,116);
Pa5 = zeros(1,116);


disp('----matching----')
tic;
for i = 1:116
    if (mod(i,50) == 0)
        disp(['query: ' num2str(i)]);
        disp(['MAP: ' num2str(mean(AP(1:i)))]);
    end
    
    % Matching
    tcnt = mod(i,200);
    if  tcnt == 1
        iu = min(N,i+200-1);
        %batchtr = tmpmat.qRefDescriptors(i:iu,:);
        batchtr = tmpmat.qRefDescriptors(1,i:iu);
        if ~ variant
            batcht = tmpmat.qDescriptors(1,i:iu);
        end
        batchta = queriesmat.refaspect(i:iu,1);
    end
    if  tcnt == 0
        tcnt = 200;
    end
    
    name = queriesInfo(i).name;
    name = string(name);
    idx = find(wnames == name);
    %trquery = batchtr(tcnt,:); %queriesmat.refDescriptors(i,:);
    tempquery = tmpmat.qRefDescriptors(1,idx);
    trquery = cell2mat(tempquery);
    qaspect = batchta(tcnt);
    %Daspect = pdist2(1/qaspect,1./wrefaspect);
    paspect = ones(size(wrefaspect))';%
    %paspect = 1.2 - .2*gaussmf(wrefaspect,[2 qaspect])';%(1+Daspect);
    
    Dist = pdist2(trquery,wrdesc);
    Dist = min(reshape(Dist,[numel(Dist)/numel(wnames) numel(wnames)]),[],1);
    
    if ~ variant
        [~,sorted_id] = sort(Dist.*paspect);
        tempqr = tmpmat.qDescriptors(1,idx);
        tquery = cell2mat(tempqr);
        %tquery = tquery*A;
        Kbest = round(percRef*numel(wnames));
        f_sorted_id = sorted_id(1:Kbest);
        
        
        wordDescBatch = wdesc(f_sorted_id);
        
        tdist = zeros(1,Kbest);
        for j = 1:Kbest
            tdesc = wordDescBatch{j};
            tdist(j) = valid_sequence_weighted_multi(tquery,tdesc,ns);
        end
        

        [~,t_sorted_id] = sort(tdist);
        %[~,t_sorted_id] = sort(tdist.*paspect(f_sorted_id));
        sorted_id = [f_sorted_id(t_sorted_id) sorted_id(Kbest+1:end)];
    else
        [~,sorted_id] = sort(Dist);
    end
    
    % Evaluate 
    relevant_list = queriesInfo(i).relevantList;
    swnames = wnames(sorted_id);
    if (~isempty(queriesInfo(i).self))
        is = ismember(swnames,queriesInfo(i).self);
        swnames(is) = [];
    end
    ids = ismember(swnames,relevant_list);

    %ids = find(ids);
    %ids = sort(ids);
    
    [AP(i),Pa5(i)] = eval_metrics(ids);
end
toc;

clear('wdesc');
disp('-----Results-----');
disp([' P@5 : ', num2str(mean(Pa5))]);
disp([' MAP : ', num2str(mean(AP))]);