relevantList = matfile('temp_BenthamICFHR14_queries/RelevantList');
queriesFound = [];
count = 0;
for i = 1:320
    query = relevantList.queriesInfo(1,i);
    relevantQueries = query.relevantList;
    common  = intersect(relevantQueries,queriesFound);
    if numel(common) > 0
        continue
    end
    relevantSample = [];
    for j = 1:numel(relevantQueries)
        a = relevantQueries(j);
        str = string(a);
        c = strsplit(str,'.');
        num = str2double(c(1));
        if num < 2501
           queriesFound = [queriesFound,a];
           relevantSample = [relevantSample,a];
        end
    end
    if numel(relevantSample) == 1
       continue 
    end
    for j = 1:numel(relevantSample)
        count = count+1;
        name = relevantSample(j);
        queriesInfo(count).name = name;
        queriesInfo(count).relevantList = relevantSample;
        queriesInfo(count).self = [];
    end
end
save('SampleRelevantList','queriesInfo')