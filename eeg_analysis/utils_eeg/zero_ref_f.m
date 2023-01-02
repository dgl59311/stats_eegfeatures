function EEG_zero = zero_ref_f(EEG_data)

    % All the functions were obtained from pop_REST_reref.m
    
	% load fixed dipoles and define their oritations
	% it can be defined by a file with dipole coordinates
	[ProgramPath, ~, ~] = fileparts(which('pop_REST_reref.m'));
	xyz_dipoles = load([ProgramPath,filesep,'corti869-3000dipoles.dat']);
    
    % channel information
    xyz_elec(:, 1) = cell2mat({EEG_data.chanlocs.X});
    xyz_elec(:, 2) = cell2mat({EEG_data.chanlocs.Y});
    xyz_elec(:, 3) = cell2mat({EEG_data.chanlocs.Z});
    
    % Parameters from pop_REST_reref.m function "REST" v1.2
	% Calculate the dipole orientations.
	xyz_dipOri           = bsxfun ( @rdivide, xyz_dipoles, sqrt ( sum ( xyz_dipoles .^ 2, 2 ) ) );
	xyz_dipOri ( 2601: 3000, 1 ) = 0;
	xyz_dipOri ( 2601: 3000, 2 ) = 0;
	xyz_dipOri ( 2601: 3000, 3 ) = 1;
	% ------------------
	% define headmodel
	headmodel        = [];
	headmodel.type   = 'concentricspheres';
	headmodel.o      = [ 0.0000 0.0000 0.0000 ];
	headmodel.r      = [ 0.8700,0.9200,1];
	headmodel.cond   = [ 1.0000,0.0125,1];
	headmodel.tissue = { 'brain' 'skull' 'scalp' };
	% -------------------
	% calculate leadfield
	[G,~] = dong_calc_leadfield3(xyz_elec,xyz_dipoles,xyz_dipOri,headmodel);
	G = G';   
    % re-ref EEG data
    [data_z] = rest_refer(EEG_data.data,G);
    
    % edit info 
    EEG_data.data = data_z;
    EEG_data.ref = 'REST';
    EEG_data.comments = 'Re-referencing to REST';
    
	[~, Origdatfile1, ext] = fileparts(EEG_data.datfile);
	EEG_data.datfile = [Origdatfile1,'_REST',ext];
   
    [~, Origdatfile2, ext] = fileparts(EEG_data.filename);
    EEG_data.filename = [Origdatfile2,'_REST',ext];
    
    [~, Origdatfile3, ext] = fileparts(EEG_data.setname);
    EEG_data.setname = [Origdatfile3,'_REST',ext];
    EEG_zero = EEG_data;
    
end