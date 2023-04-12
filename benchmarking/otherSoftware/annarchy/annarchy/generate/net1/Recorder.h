#pragma once
extern long int t;

int addRecorder(class Monitor* recorder);
Monitor* getRecorder(int id);
void removeRecorder(class Monitor* recorder);

/*
 * Recorders
 *
 */
class Monitor
{
public:
    Monitor(std::vector<int> ranks, int period, int period_offset, long int offset) {
        this->ranks = ranks;
        this->period_ = period;
        this->period_offset_ = period_offset;
        this->offset_ = offset;
        if(this->ranks.size() ==1 && this->ranks[0]==-1) // All neurons should be recorded
            this->partial = false;
        else
            this->partial = true;
    };

    ~Monitor() = default;

    virtual void record() = 0;
    virtual void record_targets() = 0;
    virtual long int size_in_bytes() = 0;
    virtual void clear() = 0;

    // Attributes
    bool partial;
    std::vector<int> ranks;
    int period_;
    int period_offset_;
    long int offset_;
};

class PopRecorder0 : public Monitor
{
protected:
    PopRecorder0(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
        {
    #ifdef _DEBUG
        std::cout << "PopRecorder0 (" << this << ") instantiated." << std::endl;
    #endif

        this->g_inh = std::vector< std::vector< double > >();
        this->record_g_inh = false; 
        this->v = std::vector< std::vector< double > >();
        this->record_v = false; 
        this->T = std::vector< std::vector< double > >();
        this->record_T = false; 
        this->r = std::vector< std::vector< double > >();
        this->record_r = false; 
        this->spike = std::map<int,  std::vector< long int > >();
        if(!this->partial){
            for(int i=0; i<pop0.size; i++) {
                this->spike[i]=std::vector<long int>();
            }
        }
        else{
            for(int i=0; i<this->ranks.size(); i++) {
                this->spike[this->ranks[i]]=std::vector<long int>();
            }
        }
        this->record_spike = false; 

    }

public:

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new PopRecorder0(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "PopRecorder0 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static PopRecorder0* get_instance(int id) {
        return static_cast<PopRecorder0*>(getRecorder(id));
    }

    void record() {

        if(this->record_v && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            cudaMemcpy(pop0.v.data(), pop0.gpu_v, pop0.size * sizeof(double), cudaMemcpyDeviceToHost);
        #ifdef _DEBUG
            auto err = cudaGetLastError();
            if ( err != cudaSuccess ) {
                std::cout << "record v on pop0 failed: " << cudaGetErrorString(err) << std::endl;
            } else {
                std::cout << "record v - [min, max]: " << *std::min_element(pop0.v.begin(), pop0.v.end() ) << ", " << *std::max_element(pop0.v.begin(), pop0.v.end() ) << std::endl;
            }
        #endif
            if(!this->partial)
                this->v.push_back(pop0.v);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop0.v[this->ranks[i]]);
                }
                this->v.push_back(tmp);
            }
        }
        if(this->record_T && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            cudaMemcpy(pop0.T.data(), pop0.gpu_T, pop0.size * sizeof(double), cudaMemcpyDeviceToHost);
        #ifdef _DEBUG
            auto err = cudaGetLastError();
            if ( err != cudaSuccess ) {
                std::cout << "record T on pop0 failed: " << cudaGetErrorString(err) << std::endl;
            } else {
                std::cout << "record T - [min, max]: " << *std::min_element(pop0.T.begin(), pop0.T.end() ) << ", " << *std::max_element(pop0.T.begin(), pop0.T.end() ) << std::endl;
            }
        #endif
            if(!this->partial)
                this->T.push_back(pop0.T);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop0.T[this->ranks[i]]);
                }
                this->T.push_back(tmp);
            }
        }
        if(this->record_r && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            cudaMemcpy(pop0.r.data(), pop0.gpu_r, pop0.size * sizeof(double), cudaMemcpyDeviceToHost);
        #ifdef _DEBUG
            auto err = cudaGetLastError();
            if ( err != cudaSuccess ) {
                std::cout << "record r on pop0 failed: " << cudaGetErrorString(err) << std::endl;
            } else {
                std::cout << "record r - [min, max]: " << *std::min_element(pop0.r.begin(), pop0.r.end() ) << ", " << *std::max_element(pop0.r.begin(), pop0.r.end() ) << std::endl;
            }
        #endif
            if(!this->partial)
                this->r.push_back(pop0.r);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop0.r[this->ranks[i]]);
                }
                this->r.push_back(tmp);
            }
        }if(this->record_spike){
            for(int i=0; i<pop0.spike_count; i++){
                if(!this->partial){
                    this->spike[pop0.spiked[i]].push_back(t);
                }
                else{
                    if( std::find(this->ranks.begin(), this->ranks.end(), pop0.spiked[i])!=this->ranks.end() ){
                        this->spike[pop0.spiked[i]].push_back(t);
                    }
                }
            }
        } 
    }

    void record_targets() {

        if(this->record_g_inh && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            cudaMemcpy(pop0.g_inh.data(), pop0.gpu_g_inh, pop0.size * sizeof(double), cudaMemcpyDeviceToHost);
        #ifdef _DEBUG
            auto err = cudaGetLastError();
            if ( err != cudaSuccess ) {
                std::cout << "record g_inh on pop0 failed: " << cudaGetErrorString(err) << std::endl;
            } else {
                std::cout << "record g_inh - [min, max]: " << *std::min_element(pop0.g_inh.begin(), pop0.g_inh.end() ) << ", " << *std::max_element(pop0.g_inh.begin(), pop0.g_inh.end() ) << std::endl;
            }
        #endif
            if(!this->partial)
                this->g_inh.push_back(pop0.g_inh);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop0.g_inh[this->ranks[i]]);
                }
                this->g_inh.push_back(tmp);
            }
        }
    }

    long int size_in_bytes() {
        std::cout << "PopMonitor::size_in_bytes(): not implemented for cuda paradigm." << std::endl;
        return 0;
    }

    void clear() {

        for(auto it = this->v.begin(); it != this->v.end(); it++)
            it->clear();
        this->v.clear();

        for(auto it = this->T.begin(); it != this->T.end(); it++)
            it->clear();
        this->T.clear();

        for(auto it = this->r.begin(); it != this->r.end(); it++)
            it->clear();
        this->r.clear();

        // TODO:
        
    }


    // Local variable g_inh
    std::vector< std::vector< double > > g_inh ;
    bool record_g_inh ; 
    // Local variable v
    std::vector< std::vector< double > > v ;
    bool record_v ; 
    // Local variable T
    std::vector< std::vector< double > > T ;
    bool record_T ; 
    // Local variable r
    std::vector< std::vector< double > > r ;
    bool record_r ; 
    // Local variable spike
    std::map<int, std::vector< long int > > spike ;
    bool record_spike ;
    void clear_spike() {
        for ( auto it = spike.begin(); it != spike.end(); it++ ) {
            it->second.clear();
            it->second.shrink_to_fit();
        }
    }

};

class PopRecorder1 : public Monitor
{
protected:
    PopRecorder1(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
        {
    #ifdef _DEBUG
        std::cout << "PopRecorder1 (" << this << ") instantiated." << std::endl;
    #endif

        this->g_inh = std::vector< std::vector< double > >();
        this->record_g_inh = false; 
        this->v = std::vector< std::vector< double > >();
        this->record_v = false; 
        this->T = std::vector< std::vector< double > >();
        this->record_T = false; 
        this->r = std::vector< std::vector< double > >();
        this->record_r = false; 
        this->spike = std::map<int,  std::vector< long int > >();
        if(!this->partial){
            for(int i=0; i<pop1.size; i++) {
                this->spike[i]=std::vector<long int>();
            }
        }
        else{
            for(int i=0; i<this->ranks.size(); i++) {
                this->spike[this->ranks[i]]=std::vector<long int>();
            }
        }
        this->record_spike = false; 

    }

public:

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new PopRecorder1(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "PopRecorder1 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static PopRecorder1* get_instance(int id) {
        return static_cast<PopRecorder1*>(getRecorder(id));
    }

    void record() {

        if(this->record_v && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            cudaMemcpy(pop1.v.data(), pop1.gpu_v, pop1.size * sizeof(double), cudaMemcpyDeviceToHost);
        #ifdef _DEBUG
            auto err = cudaGetLastError();
            if ( err != cudaSuccess ) {
                std::cout << "record v on pop1 failed: " << cudaGetErrorString(err) << std::endl;
            } else {
                std::cout << "record v - [min, max]: " << *std::min_element(pop1.v.begin(), pop1.v.end() ) << ", " << *std::max_element(pop1.v.begin(), pop1.v.end() ) << std::endl;
            }
        #endif
            if(!this->partial)
                this->v.push_back(pop1.v);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop1.v[this->ranks[i]]);
                }
                this->v.push_back(tmp);
            }
        }
        if(this->record_T && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            cudaMemcpy(pop1.T.data(), pop1.gpu_T, pop1.size * sizeof(double), cudaMemcpyDeviceToHost);
        #ifdef _DEBUG
            auto err = cudaGetLastError();
            if ( err != cudaSuccess ) {
                std::cout << "record T on pop1 failed: " << cudaGetErrorString(err) << std::endl;
            } else {
                std::cout << "record T - [min, max]: " << *std::min_element(pop1.T.begin(), pop1.T.end() ) << ", " << *std::max_element(pop1.T.begin(), pop1.T.end() ) << std::endl;
            }
        #endif
            if(!this->partial)
                this->T.push_back(pop1.T);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop1.T[this->ranks[i]]);
                }
                this->T.push_back(tmp);
            }
        }
        if(this->record_r && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            cudaMemcpy(pop1.r.data(), pop1.gpu_r, pop1.size * sizeof(double), cudaMemcpyDeviceToHost);
        #ifdef _DEBUG
            auto err = cudaGetLastError();
            if ( err != cudaSuccess ) {
                std::cout << "record r on pop1 failed: " << cudaGetErrorString(err) << std::endl;
            } else {
                std::cout << "record r - [min, max]: " << *std::min_element(pop1.r.begin(), pop1.r.end() ) << ", " << *std::max_element(pop1.r.begin(), pop1.r.end() ) << std::endl;
            }
        #endif
            if(!this->partial)
                this->r.push_back(pop1.r);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop1.r[this->ranks[i]]);
                }
                this->r.push_back(tmp);
            }
        }if(this->record_spike){
            for(int i=0; i<pop1.spike_count; i++){
                if(!this->partial){
                    this->spike[pop1.spiked[i]].push_back(t);
                }
                else{
                    if( std::find(this->ranks.begin(), this->ranks.end(), pop1.spiked[i])!=this->ranks.end() ){
                        this->spike[pop1.spiked[i]].push_back(t);
                    }
                }
            }
        } 
    }

    void record_targets() {

        if(this->record_g_inh && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            cudaMemcpy(pop1.g_inh.data(), pop1.gpu_g_inh, pop1.size * sizeof(double), cudaMemcpyDeviceToHost);
        #ifdef _DEBUG
            auto err = cudaGetLastError();
            if ( err != cudaSuccess ) {
                std::cout << "record g_inh on pop1 failed: " << cudaGetErrorString(err) << std::endl;
            } else {
                std::cout << "record g_inh - [min, max]: " << *std::min_element(pop1.g_inh.begin(), pop1.g_inh.end() ) << ", " << *std::max_element(pop1.g_inh.begin(), pop1.g_inh.end() ) << std::endl;
            }
        #endif
            if(!this->partial)
                this->g_inh.push_back(pop1.g_inh);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop1.g_inh[this->ranks[i]]);
                }
                this->g_inh.push_back(tmp);
            }
        }
    }

    long int size_in_bytes() {
        std::cout << "PopMonitor::size_in_bytes(): not implemented for cuda paradigm." << std::endl;
        return 0;
    }

    void clear() {

        for(auto it = this->v.begin(); it != this->v.end(); it++)
            it->clear();
        this->v.clear();

        for(auto it = this->T.begin(); it != this->T.end(); it++)
            it->clear();
        this->T.clear();

        for(auto it = this->r.begin(); it != this->r.end(); it++)
            it->clear();
        this->r.clear();

        // TODO:
        
    }


    // Local variable g_inh
    std::vector< std::vector< double > > g_inh ;
    bool record_g_inh ; 
    // Local variable v
    std::vector< std::vector< double > > v ;
    bool record_v ; 
    // Local variable T
    std::vector< std::vector< double > > T ;
    bool record_T ; 
    // Local variable r
    std::vector< std::vector< double > > r ;
    bool record_r ; 
    // Local variable spike
    std::map<int, std::vector< long int > > spike ;
    bool record_spike ;
    void clear_spike() {
        for ( auto it = spike.begin(); it != spike.end(); it++ ) {
            it->second.clear();
            it->second.shrink_to_fit();
        }
    }

};

class PopRecorder2 : public Monitor
{
protected:
    PopRecorder2(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
        {
    #ifdef _DEBUG
        std::cout << "PopRecorder2 (" << this << ") instantiated." << std::endl;
    #endif

        this->g_inh = std::vector< std::vector< double > >();
        this->record_g_inh = false; 
        this->v = std::vector< std::vector< double > >();
        this->record_v = false; 
        this->T = std::vector< std::vector< double > >();
        this->record_T = false; 
        this->r = std::vector< std::vector< double > >();
        this->record_r = false; 
        this->spike = std::map<int,  std::vector< long int > >();
        if(!this->partial){
            for(int i=0; i<pop2.size; i++) {
                this->spike[i]=std::vector<long int>();
            }
        }
        else{
            for(int i=0; i<this->ranks.size(); i++) {
                this->spike[this->ranks[i]]=std::vector<long int>();
            }
        }
        this->record_spike = false; 

    }

public:

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new PopRecorder2(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "PopRecorder2 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static PopRecorder2* get_instance(int id) {
        return static_cast<PopRecorder2*>(getRecorder(id));
    }

    void record() {

        if(this->record_v && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            cudaMemcpy(pop2.v.data(), pop2.gpu_v, pop2.size * sizeof(double), cudaMemcpyDeviceToHost);
        #ifdef _DEBUG
            auto err = cudaGetLastError();
            if ( err != cudaSuccess ) {
                std::cout << "record v on pop2 failed: " << cudaGetErrorString(err) << std::endl;
            } else {
                std::cout << "record v - [min, max]: " << *std::min_element(pop2.v.begin(), pop2.v.end() ) << ", " << *std::max_element(pop2.v.begin(), pop2.v.end() ) << std::endl;
            }
        #endif
            if(!this->partial)
                this->v.push_back(pop2.v);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop2.v[this->ranks[i]]);
                }
                this->v.push_back(tmp);
            }
        }
        if(this->record_T && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            cudaMemcpy(pop2.T.data(), pop2.gpu_T, pop2.size * sizeof(double), cudaMemcpyDeviceToHost);
        #ifdef _DEBUG
            auto err = cudaGetLastError();
            if ( err != cudaSuccess ) {
                std::cout << "record T on pop2 failed: " << cudaGetErrorString(err) << std::endl;
            } else {
                std::cout << "record T - [min, max]: " << *std::min_element(pop2.T.begin(), pop2.T.end() ) << ", " << *std::max_element(pop2.T.begin(), pop2.T.end() ) << std::endl;
            }
        #endif
            if(!this->partial)
                this->T.push_back(pop2.T);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop2.T[this->ranks[i]]);
                }
                this->T.push_back(tmp);
            }
        }
        if(this->record_r && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            cudaMemcpy(pop2.r.data(), pop2.gpu_r, pop2.size * sizeof(double), cudaMemcpyDeviceToHost);
        #ifdef _DEBUG
            auto err = cudaGetLastError();
            if ( err != cudaSuccess ) {
                std::cout << "record r on pop2 failed: " << cudaGetErrorString(err) << std::endl;
            } else {
                std::cout << "record r - [min, max]: " << *std::min_element(pop2.r.begin(), pop2.r.end() ) << ", " << *std::max_element(pop2.r.begin(), pop2.r.end() ) << std::endl;
            }
        #endif
            if(!this->partial)
                this->r.push_back(pop2.r);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop2.r[this->ranks[i]]);
                }
                this->r.push_back(tmp);
            }
        }if(this->record_spike){
            for(int i=0; i<pop2.spike_count; i++){
                if(!this->partial){
                    this->spike[pop2.spiked[i]].push_back(t);
                }
                else{
                    if( std::find(this->ranks.begin(), this->ranks.end(), pop2.spiked[i])!=this->ranks.end() ){
                        this->spike[pop2.spiked[i]].push_back(t);
                    }
                }
            }
        } 
    }

    void record_targets() {

        if(this->record_g_inh && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            cudaMemcpy(pop2.g_inh.data(), pop2.gpu_g_inh, pop2.size * sizeof(double), cudaMemcpyDeviceToHost);
        #ifdef _DEBUG
            auto err = cudaGetLastError();
            if ( err != cudaSuccess ) {
                std::cout << "record g_inh on pop2 failed: " << cudaGetErrorString(err) << std::endl;
            } else {
                std::cout << "record g_inh - [min, max]: " << *std::min_element(pop2.g_inh.begin(), pop2.g_inh.end() ) << ", " << *std::max_element(pop2.g_inh.begin(), pop2.g_inh.end() ) << std::endl;
            }
        #endif
            if(!this->partial)
                this->g_inh.push_back(pop2.g_inh);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop2.g_inh[this->ranks[i]]);
                }
                this->g_inh.push_back(tmp);
            }
        }
    }

    long int size_in_bytes() {
        std::cout << "PopMonitor::size_in_bytes(): not implemented for cuda paradigm." << std::endl;
        return 0;
    }

    void clear() {

        for(auto it = this->v.begin(); it != this->v.end(); it++)
            it->clear();
        this->v.clear();

        for(auto it = this->T.begin(); it != this->T.end(); it++)
            it->clear();
        this->T.clear();

        for(auto it = this->r.begin(); it != this->r.end(); it++)
            it->clear();
        this->r.clear();

        // TODO:
        
    }


    // Local variable g_inh
    std::vector< std::vector< double > > g_inh ;
    bool record_g_inh ; 
    // Local variable v
    std::vector< std::vector< double > > v ;
    bool record_v ; 
    // Local variable T
    std::vector< std::vector< double > > T ;
    bool record_T ; 
    // Local variable r
    std::vector< std::vector< double > > r ;
    bool record_r ; 
    // Local variable spike
    std::map<int, std::vector< long int > > spike ;
    bool record_spike ;
    void clear_spike() {
        for ( auto it = spike.begin(); it != spike.end(); it++ ) {
            it->second.clear();
            it->second.shrink_to_fit();
        }
    }

};

class PopRecorder3 : public Monitor
{
protected:
    PopRecorder3(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
        {
    #ifdef _DEBUG
        std::cout << "PopRecorder3 (" << this << ") instantiated." << std::endl;
    #endif

        this->g_inh = std::vector< std::vector< double > >();
        this->record_g_inh = false; 
        this->v = std::vector< std::vector< double > >();
        this->record_v = false; 
        this->T = std::vector< std::vector< double > >();
        this->record_T = false; 
        this->r = std::vector< std::vector< double > >();
        this->record_r = false; 
        this->spike = std::map<int,  std::vector< long int > >();
        if(!this->partial){
            for(int i=0; i<pop3.size; i++) {
                this->spike[i]=std::vector<long int>();
            }
        }
        else{
            for(int i=0; i<this->ranks.size(); i++) {
                this->spike[this->ranks[i]]=std::vector<long int>();
            }
        }
        this->record_spike = false; 

    }

public:

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new PopRecorder3(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "PopRecorder3 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static PopRecorder3* get_instance(int id) {
        return static_cast<PopRecorder3*>(getRecorder(id));
    }

    void record() {

        if(this->record_v && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            cudaMemcpy(pop3.v.data(), pop3.gpu_v, pop3.size * sizeof(double), cudaMemcpyDeviceToHost);
        #ifdef _DEBUG
            auto err = cudaGetLastError();
            if ( err != cudaSuccess ) {
                std::cout << "record v on pop3 failed: " << cudaGetErrorString(err) << std::endl;
            } else {
                std::cout << "record v - [min, max]: " << *std::min_element(pop3.v.begin(), pop3.v.end() ) << ", " << *std::max_element(pop3.v.begin(), pop3.v.end() ) << std::endl;
            }
        #endif
            if(!this->partial)
                this->v.push_back(pop3.v);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop3.v[this->ranks[i]]);
                }
                this->v.push_back(tmp);
            }
        }
        if(this->record_T && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            cudaMemcpy(pop3.T.data(), pop3.gpu_T, pop3.size * sizeof(double), cudaMemcpyDeviceToHost);
        #ifdef _DEBUG
            auto err = cudaGetLastError();
            if ( err != cudaSuccess ) {
                std::cout << "record T on pop3 failed: " << cudaGetErrorString(err) << std::endl;
            } else {
                std::cout << "record T - [min, max]: " << *std::min_element(pop3.T.begin(), pop3.T.end() ) << ", " << *std::max_element(pop3.T.begin(), pop3.T.end() ) << std::endl;
            }
        #endif
            if(!this->partial)
                this->T.push_back(pop3.T);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop3.T[this->ranks[i]]);
                }
                this->T.push_back(tmp);
            }
        }
        if(this->record_r && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            cudaMemcpy(pop3.r.data(), pop3.gpu_r, pop3.size * sizeof(double), cudaMemcpyDeviceToHost);
        #ifdef _DEBUG
            auto err = cudaGetLastError();
            if ( err != cudaSuccess ) {
                std::cout << "record r on pop3 failed: " << cudaGetErrorString(err) << std::endl;
            } else {
                std::cout << "record r - [min, max]: " << *std::min_element(pop3.r.begin(), pop3.r.end() ) << ", " << *std::max_element(pop3.r.begin(), pop3.r.end() ) << std::endl;
            }
        #endif
            if(!this->partial)
                this->r.push_back(pop3.r);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop3.r[this->ranks[i]]);
                }
                this->r.push_back(tmp);
            }
        }if(this->record_spike){
            for(int i=0; i<pop3.spike_count; i++){
                if(!this->partial){
                    this->spike[pop3.spiked[i]].push_back(t);
                }
                else{
                    if( std::find(this->ranks.begin(), this->ranks.end(), pop3.spiked[i])!=this->ranks.end() ){
                        this->spike[pop3.spiked[i]].push_back(t);
                    }
                }
            }
        } 
    }

    void record_targets() {

        if(this->record_g_inh && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            cudaMemcpy(pop3.g_inh.data(), pop3.gpu_g_inh, pop3.size * sizeof(double), cudaMemcpyDeviceToHost);
        #ifdef _DEBUG
            auto err = cudaGetLastError();
            if ( err != cudaSuccess ) {
                std::cout << "record g_inh on pop3 failed: " << cudaGetErrorString(err) << std::endl;
            } else {
                std::cout << "record g_inh - [min, max]: " << *std::min_element(pop3.g_inh.begin(), pop3.g_inh.end() ) << ", " << *std::max_element(pop3.g_inh.begin(), pop3.g_inh.end() ) << std::endl;
            }
        #endif
            if(!this->partial)
                this->g_inh.push_back(pop3.g_inh);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop3.g_inh[this->ranks[i]]);
                }
                this->g_inh.push_back(tmp);
            }
        }
    }

    long int size_in_bytes() {
        std::cout << "PopMonitor::size_in_bytes(): not implemented for cuda paradigm." << std::endl;
        return 0;
    }

    void clear() {

        for(auto it = this->v.begin(); it != this->v.end(); it++)
            it->clear();
        this->v.clear();

        for(auto it = this->T.begin(); it != this->T.end(); it++)
            it->clear();
        this->T.clear();

        for(auto it = this->r.begin(); it != this->r.end(); it++)
            it->clear();
        this->r.clear();

        // TODO:
        
    }


    // Local variable g_inh
    std::vector< std::vector< double > > g_inh ;
    bool record_g_inh ; 
    // Local variable v
    std::vector< std::vector< double > > v ;
    bool record_v ; 
    // Local variable T
    std::vector< std::vector< double > > T ;
    bool record_T ; 
    // Local variable r
    std::vector< std::vector< double > > r ;
    bool record_r ; 
    // Local variable spike
    std::map<int, std::vector< long int > > spike ;
    bool record_spike ;
    void clear_spike() {
        for ( auto it = spike.begin(); it != spike.end(); it++ ) {
            it->second.clear();
            it->second.shrink_to_fit();
        }
    }

};

class ProjRecorder0 : public Monitor
{
protected:
    ProjRecorder0(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset)
    {
    #ifdef _DEBUG
        std::cout << "ProjRecorder0 (" << this << ") instantiated." << std::endl;
    #endif

    };

public:

    static int create_instance(std::vector<int> ranks, int period, int period_offset, long int offset) {
        auto new_recorder = new ProjRecorder0(ranks, period, period_offset, offset);
        auto id = addRecorder(static_cast<Monitor*>(new_recorder));
    #ifdef _DEBUG
        std::cout << "ProjRecorder0 (" << new_recorder << ") received list position (ID) = " << id << std::endl;
    #endif
        return id;
    }

    static ProjRecorder0* get_instance(int id) {
        return static_cast<ProjRecorder0*>(getRecorder(id));
    }

    void record() {

    };

    void record_targets() { /* nothing to do here */ }
    long int size_in_bytes() {
        std::cout << "ProjMonitor::size_in_bytes(): not implemented for cuda paradigm." << std::endl;
        return 0;
    }

    void clear() {
        std::cout << "ProjRecorder0::clear(): not implemented for cuda paradigm." << std::endl;
    }


};

