M = dlmread('vae_low_dimension_vectors.txt');
wmatname = 'words_POG_Sequential';
wordsmat = matfile(['temp_descriptors/' wmatname]);
wnames = wordsmat.names;
for i = 1:2501
    words.name(i) = wnames(i);
    words.des(i,:) = M(i,:);
end
queries = matfile('SampleRelevantList');
N = numel(queries.queriesInfo);
AP = zeros(1,N);
Pa5 = zeros(1,N);
wdesc = words.des;
queryInfo = queries.queriesInfo;
for i = 1:N
    name = queryInfo(i).name;
    name = string(name);
    idx = find(wnames == name);
    query = words.des(idx,:);
    Dist = pdist2(query,wdesc);
    [~,sorted_id] = sort(Dist);
    relevantList = queryInfo(i).relevantList;
    swnames = wnames(sorted_id);
    if (~isempty(queryInfo(i).self))
        is = ismember(swnames,queriesInfo(ii).self);
        swnames(is) = [];
    end
    ids = ismember(swnames,relevantList);
    [AP(i),Pa5(i)] = eval_metrics(ids);
end
disp('-----Results-----');
    disp([' P@5 : ', num2str(mean(Pa5))]);
    disp([' MAP : ', num2str(mean(AP))]);
%}