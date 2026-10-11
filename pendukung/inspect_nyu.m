file = fullfile(pwd, 'data', 'nyu_depth_v2_labeled.mat');
info = h5info(file);
fprintf('type: MATLAB v7.3 MAT-file (HDF5)\n');
fprintf('root_groups: %s\n', strjoin({info.Groups.Name}, ', '));
fprintf('root_datasets: %s\n', strjoin({info.Datasets.Name}, ', '));
function walk(g, prefix)
  for i=1:numel(g.Datasets)
    d=g.Datasets(i); fprintf('dataset: %s shape=[%s] dtype=%s\n', d.Name, sprintf('%d ', d.Dataspace.Size), d.Datatype.Class);
  end
  for i=1:numel(g.Groups), walk(g.Groups(i), prefix); end
end
walk(info, '');
for i=1:numel(info.Datasets)
  d=info.Datasets(i); n=d.Name;
  if contains(lower(n),'image') || contains(lower(n),'depth')
    fprintf('target: %s shape=[%s] dtype=%s\n', n, sprintf('%d ', d.Dataspace.Size), d.Datatype.Class);
    sz=double(d.Dataspace.Size); nd=numel(sz); chunk=sz; chunk(1)=min(sz(1),8); st=ones(1,nd); ct=chunk;
    try
      x=h5read(file,n,st,ct); x=double(x); finite=x(isfinite(x)); fprintf('sample_range: [%g, %g]\n', min(finite), max(finite));
    catch e, fprintf('sample_error: %s\n',e.message); end
  end
end
