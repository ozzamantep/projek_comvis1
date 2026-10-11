file = fullfile(pwd, 'data', 'nyu_depth_v2_labeled.mat');
for n={'/depths','/images'}
 p=n{1}; info=h5info(file,p); fprintf('%s shape=[%s] dtype=%s\n',p,sprintf('%d ',info.Dataspace.Size),info.Datatype.Class);
 sz=double(info.Dataspace.Size); st=ones(1,numel(sz)); ct=sz; ct(end)=min(sz(end),8);
 x=double(h5read(file,p,st,ct)); q=x(isfinite(x)); fprintf('%s sample_range=[%g, %g], sample_count=%d\n',p,min(q),max(q),numel(q));
end
