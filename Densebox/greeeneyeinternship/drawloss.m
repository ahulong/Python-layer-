root_dirs={...
    '/home/panzheng/Public/TitanX01/caffe/models/ClsRegNet',...
    '/home/panzheng/Public/TitanX01/caffe/models/ClsRegNet',...
    '/home/panzheng/Public/TitanX01/caffe/models/ClsRegNet',...
    };

fignames={'10layer_morenight'};



epoch_x=false;

for i=1:length(fignames)
   
    folder=sprintf('%s/%s/',root_dirs{i},fignames{i});

    [test_out,test_iter,train_loss,train_iters,train_out]=parse_loss(folder,0);

    
    if epoch_x
        test_iter = test_iter*256/1024/749;
        train_iters = train_iters*256/1024/749;
    end
    
    figure(i);
    plot(train_iters,train_loss);    hold on;
    
    plot(test_iter(1:size(test_out,1)),test_out(:,1),'r','linewidth',2);
    plot(test_iter(1:size(test_out,1)),test_out(:,2),'c','linewidth',2);
    plot(test_iter(1:size(test_out,1)),test_out(:,3),'m','linewidth',2);
    
    grid on;
    hold off;
    legend({'train','test #0','test #1'},2);
    xlabel(fignames{i},'interpreter','none');
    ylim([0.93,0.97]);
%     xlim([0 5e5]);
%     set(gca,'yscale','log')

    fprintf('%s\n',fignames{i});
end