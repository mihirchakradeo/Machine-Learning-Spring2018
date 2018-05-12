%=========================================================================   
%Solutions for Question 4

%Solution to Question 4.4.1

%Uncomment this for using Quadratic SVM
%[trD, trLb, valD, valLb, trRegs, valRegs] = HW2_Utils.getPosAndRandomNeg();
%[W, b] = SVMQuadratic(trD, trLb);
%HW2_Utils.genRsltFile(W, b, "val", "resultFile");
%[ap, prec, rec] = HW2_Utils.cmpAP("resultFile", "val");


%Uncomment this to use the SGD SVM
%W = SGD(trD,trLb);
%W_2 = SGD2(W, valD, valLb);

%HW2_Utils.genRsl9tFile(W, 0, "val", "resultFile");
%[ap, prec, rec] = HW2_Utils.cmpAP("resultFile", "val")
%=========================================================================   

dataDir = '../hw2data';
%Solution to 4.4.3
%[W_trained, b_trained, f_0_array] = hard_negative_mining_algorithm;
%HW2_Utils.genRsltFile(W_trained, b_trained, "test", "1114621882");
                
function [W, b, f_0_array, ap_array] = hard_negative_mining_algorithm()
    %generating positive and negative training data
    [trD, trLb, valD, valLb, trRegs, valRegs] = HW2_Utils.getPosAndRandomNeg();
    f_0_array = [];
    ap_array = [];
    %Extracting negative numbers from the generated data
    
    %PosD all annotated upper bodies, NegD random image patches
    PosD = trD(:, trLb>0);
    PosD_Label = trLb(trLb>0);
    NegD = trD(:, trLb<0); %dimension: 1984x186
    A = trLb(trLb<0);
    
    [W, b, f_0] = SVMQuadratic(trD, trLb); 
    load(sprintf('%s/%sAnno.mat', HW2_Utils.dataDir, "train"), 'ubAnno');

    %Give valid stopping condition here, right now setting static value
    for iter = 1:10
        fprintf("Iteration %d",iter);
        temp_PosD_Label = PosD_Label;
        %Step 1: Getting all non support vectors in NegD
        temp1 = W'*NegD;
        temp1 = temp1(1,:);
        
        temp2 = (temp1<=1 & temp1>=-1);
        A = NegD(:, temp2);
        
        %[t,lt_m1] = size(temp1(temp1<-1));
        %[t,gt_1] = size(temp1(temp1>1));
        %lt_m1+gt_1
        
        
        %STEP 2: detect(im, w, b, shldDisplay)
        %Getting images (im)


        rects = cell(1, 93);
        for i=1:93
            im = sprintf('%s/trainIms/%04d.jpg', HW2_Utils.dataDir, i);
            im = imread(im);
            [imH, imW,~] = size(im);
            rects{i} = HW2_Utils.detect(im, W, b);
            
            %Getting index from where the negative numbers start
            
            number_of_negative = sum(rects{i}(end, :)<0);
            number_of_positive = size(rects{i}(1,:),2) - number_of_negative;
            
            if number_of_negative == 0
                continue
            end
            if number_of_positive == 0
                number_of_positive = number_of_positive + 1;
            end
            
            arr = rects{i}(:,end-number_of_negative+1:end);
            pos_arr = cell2mat(ubAnno(i));
            
            % Remove random rects that do not lie within image boundaries
            badIdxs = or(arr(3,:) > imW, arr(4,:) > imH);
            arr_n = arr(:,~badIdxs);
            arr_n;
            
            % Remove random rects that overlap more than 30% with an annotated upper body
            for j=1:size(arr_n,2)
                overlap = HW2_Utils.rectOverlap(arr_n, pos_arr);                    
                arr_n = arr_n(:, overlap < 0.001);
                if isempty(arr_n)
                    break;
                end;
            end;
            %arr_n
            
            % Feature Extraction
            [D_i, R_i] = deal(cell(1, size(arr_n,2)));
            for j=1:length(D_i)
                ub = arr_n(:,j);
                imReg = im(ub(2):ub(4), ub(1):ub(3),:);
                imReg = imresize(imReg, HW2_Utils.normImSz);
                D_i{j} = HW2_Utils.cmpFeat(rgb2gray(imReg));
                R_i{j} = imReg;
            end 
            mat = cell2mat(D_i);
            temp_size = size(mat,2);
            total_temp_size = size(NegD,2) + temp_size;
            
            if total_temp_size > 1000*iter
                break
            end
            
            NegD = horzcat(NegD, mat);
            size(NegD);
            
        end
        
        %Step 3: Retrain
        temp_trD = horzcat(PosD, NegD);
        temp_ones = -ones(size(NegD,2), 1);
        size(temp_ones);
        size(temp_PosD_Label);
        temp_trLb = vertcat(temp_PosD_Label, temp_ones);
        size(temp_trLb);
        [W, b, f_0] = SVMQuadratic(temp_trD, temp_trLb);
                
        %Getting the objective values
        f_0_array = [f_0_array;f_0]
        
        %Getting the ap values
        HW2_Utils.genRsltFile(W, b, "val", "temp_val");
        [ap, prec, rec] = HW2_Utils.cmpAP("temp_val", "val");
        ap_array = [ap_array;ap]
        
        
    end
end
    


function result = predict(W, X)
    result = X'*W;
end


function [W,dict,dict2] = SGD(trD, trLb)

    [d,n] = size(trD);
    k = length(unique(trLb));

    %mapping labels to 1...k
    keyset = [1:k];
    valueset = unique(trLb);
    dict = containers.Map(valueset, keyset);
    dict2 = containers.Map(keyset, valueset);
    
    max_epoch = 100;
    epoch = 1;
    eta_0 = 1;
    eta_1 = 100;
    eta = 0;
    C = 0.1; 
    W = zeros(d,k);

    loss_per_epoch = [];

    for epoch = 1:max_epoch
        loss_per_iteration = 0;
        eta = eta_0/(eta_1+epoch);
        temp = randperm(n);
        for i = temp

            loss_term_1 = 0;

            yi_hat_arr = [];

            %first calculating yi
            yi_temp = trLb(i);
            yi = dict(yi_temp);

            %calculating yi_hat
            for j = 1:k
                if j == yi
                    yi_hat_arr = [yi_hat_arr;-Inf];
                    continue
                else
                    yi_hat_temp = W(:,j)'*trD(:,i);
                    yi_hat_arr = [yi_hat_arr;yi_hat_temp];
                end
            end
            [val,yi_hat] = max(yi_hat_arr);
            yi_hat;

            %check if yi and yi_hat are same or not
            if yi == yi_hat
               disp("SAME yi and yi_hat")
            end

            derivative = (1/n)*W;

            if W(:,yi_hat)'*trD(:,i) - W(:,yi)'*trD(:,i)+1>0
                %y1
                derivative(:,yi) = derivative(:,yi) - C*trD(:,i);
                %y1_hat
                derivative(:,yi_hat) = derivative(:,yi_hat) + C*trD(:,i);
            end
            W = W - eta*derivative;

            %calculating loss
            for j=1:k
                loss_term_1 = loss_term_1 + norm(W(:,j),2)^2;
            end
            loss_term_1 = (1/(2*n))*loss_term_1;
            second_term = C*max((W(:,yi_hat)'*trD(:,i)-W(:,yi)'*trD(:,i)+1),0);
            loss_per_iteration = loss_per_iteration + loss_term_1 + second_term;
        end
        loss_per_epoch = [loss_per_epoch;loss_per_iteration];
    end
    %plot(loss_per_epoch)
end %function end


function [W,dict,dict2] = SGD2(W, trD, trLb)

    [d,n] = size(trD);
    k = length(unique(trLb));

    %mapping labels to 1...k
    keyset = [1:k];
    valueset = unique(trLb);
    dict = containers.Map(valueset, keyset);
    dict2 = containers.Map(keyset, valueset);
    
    max_epoch = 2000;
    epoch = 1;
    eta_0 = 1;
    eta_1 = 100;
    eta = 0;
    C = 0.0001; 
    %W = zeros(d,k);

    loss_per_epoch = [];

    for epoch = 1:max_epoch
        loss_per_iteration = 0;
        eta = eta_0/(eta_1+epoch);
        temp = randperm(n);
        for i = temp

            loss_term_1 = 0;

            yi_hat_arr = [];

            %first calculating yi
            yi_temp = trLb(i);
            yi = dict(yi_temp);

            %calculating yi_hat
            for j = 1:k
                if j == yi
                    yi_hat_arr = [yi_hat_arr;-Inf];
                    continue
                else
                    yi_hat_temp = W(:,j)'*trD(:,i);
                    yi_hat_arr = [yi_hat_arr;yi_hat_temp];
                end
            end
            [val,yi_hat] = max(yi_hat_arr);
            yi_hat;

            %check if yi and yi_hat are same or not
            if yi == yi_hat
               disp("SAME yi and yi_hat")
            end

            derivative = (1/n)*W;

            if W(:,yi_hat)'*trD(:,i) - W(:,yi)'*trD(:,i)+1>0
                %y1
                derivative(:,yi) = derivative(:,yi) - C*trD(:,i);
                %y1_hat
                derivative(:,yi_hat) = derivative(:,yi_hat) + C*trD(:,i);
            end
            W = W - eta*derivative;

            %calculating loss
            for j=1:k
                loss_term_1 = loss_term_1 + norm(W(:,j),2)^2;
            end
            loss_term_1 = (1/(2*n))*loss_term_1;
            second_term = C*max((W(:,yi_hat)'*trD(:,i)-W(:,yi)'*trD(:,i)+1),0);
            loss_per_iteration = loss_per_iteration + loss_term_1 + second_term;
        end
        loss_per_epoch = [loss_per_epoch;loss_per_iteration];
    end
    %plot(loss_per_epoch)
end %function end


function [W, b, f_0] = SVMQuadratic(trD, trLb)
    trD = double(trD);
    trLb = double(trLb);
    C = 10;
    [d,n] = size(trD);
    k = trD'*trD;
    H = diag(trLb)*k*diag(trLb);
    f = -ones(1,n);
    A = zeros(1,n);
    b = 0;
    Aeq = trLb';
    beq = 0;
    lb = zeros(n,1);
    ub = C*ones(n,1);

    [alpha, f_0] = quadprog(H,f,A,b,Aeq,beq,lb,ub);
    f_0 = -f_0;

    %Solution to question 3.1.3 and 3.1.4
    X = trD';

    alpha_n = diag(alpha);
    W = (trLb'*alpha_n*X)';
    b = trLb - trD'*W;
    b = b(alpha>0 & alpha<C);
    b = min(b)/2;

end


