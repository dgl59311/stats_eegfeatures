function [matrices] = read_connectivity(file_dir)
% read_connectivity reads text files from Loreta

    for nmat = 1:length(file_dir)
    
        fid = fopen(fullfile(file_dir(nmat).folder, file_dir(nmat).name));
        tline = fgetl(fid);
        tlines = cell(0,1);
    
        while ischar(tline)
            tlines{end+1,1} = tline;
            tline = fgetl(fid);
        end
        fclose(fid);
        all_files(:,nmat) = tlines(:,1);
    end
    
    em_ = ~all(cellfun('isempty', all_files),2);
    cuts_ = find(em_==0);
    cuts_ = [cuts_;size(all_files, 1)+1];

    for c_ = 1:(length(cuts_)-1)   
        
        for nmat = 1:length(file_dir)
    
            ulim = cuts_(c_)+1;
            llim = cuts_(c_+1)-1;
            cnx = str2double(split...
                     (all_files(ulim:llim,nmat)));
            cnx = cnx(:,2:end);
            matrices(nmat,c_,:,:) = cnx;
   
        end
    end
end