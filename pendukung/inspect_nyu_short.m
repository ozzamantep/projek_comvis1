file = fullfile(pwd, 'data', 'nyu_depth_v2_labeled.mat');
info=h5info(file);
fprintf('type: MATLAB v7.3 MAT-file (HDF5)\n');
fprintf('root_groups: %s\n', strjoin({info.Groups.Name}, ', '));
fprintf('root_datasets: %s\n', strjoin({info.Datasets.Name}, ', '));
scan(info,file);
function scan(g,file)
 for i=1:numel(g.Datasets)
  d=g.Datasets(i); n=d.Name; low=lower(n);
  if contains(low,'image') || contains(low,'depth') || contains(low,'name')
   fprintf('dataset: %s shape=[%s] dtype=%s\n',n,sprintf('%d ',d.Dataspace.Size),d.Datatype.Class);
   sz=double(d.Dataspace.Size); st=ones(1,numel(sz)); ct=sz; ct(1)=min(sz(1),8);
   try
    x=double(h5read(file,n,st,ct)); q=x(isfinite(x));
    fprintf('sample_range: [%g, %g] sample_count=%d\n',min(q),max(q),numel(q));
   catch e, fprintf('sample_error: %s\n',e.message); end
  end
 end
 for i=1:numel(g.Groups), scan(g.Groups(i),file); end
end
