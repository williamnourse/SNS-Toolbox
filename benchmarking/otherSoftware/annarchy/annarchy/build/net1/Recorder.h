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

        this->_sum_inh = std::vector< std::vector< double > >();
        this->record__sum_inh = false; 
        this->v = std::vector< std::vector< double > >();
        this->record_v = false; 
        this->r = std::vector< std::vector< double > >();
        this->record_r = false; 
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
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "PopRecorder0::record()" << std::endl;
    #endif

        if(this->record_v && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
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
        if(this->record_r && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->r.push_back(pop0.r);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop0.r[this->ranks[i]]);
                }
                this->r.push_back(tmp);
            }
        }
    }

    void record_targets() {

        if(this->record__sum_inh && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            if(!this->partial)
                this->_sum_inh.push_back(pop0._sum_inh);
            else{
                std::vector<double> tmp = std::vector<double>();
                for (unsigned int i=0; i<this->ranks.size(); i++){
                    tmp.push_back(pop0._sum_inh[this->ranks[i]]);
                }
                this->_sum_inh.push_back(tmp);
            }
        }
    }

    long int size_in_bytes() {
        long int size_in_bytes = 0;
        
        // local variable v
        size_in_bytes += sizeof(std::vector<double>) * v.capacity();
        for(auto it=v.begin(); it!= v.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        
        // local variable r
        size_in_bytes += sizeof(std::vector<double>) * r.capacity();
        for(auto it=r.begin(); it!= r.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(double);
        }
        
        return size_in_bytes;
    }

    void clear() {
    #ifdef _DEBUG
        std::cout << "Delete instance of PopRecorder0 ( " << this << " ) " << std::endl;
    #endif

        for(auto it = this->v.begin(); it != this->v.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->v.clear();
    
        for(auto it = this->r.begin(); it != this->r.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->r.clear();
    

        removeRecorder(this);
    }



    // Local variable _sum_inh
    std::vector< std::vector< double > > _sum_inh ;
    bool record__sum_inh ; 
    // Local variable v
    std::vector< std::vector< double > > v ;
    bool record_v ; 
    // Local variable r
    std::vector< std::vector< double > > r ;
    bool record_r ; 
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
        std::map< int, int > post_indices = std::map< int, int > ();
        auto post_rank = proj0.get_post_rank();

        for(int i=0; i<post_rank.size(); i++){
            post_indices[post_rank[i]] = i;
        }
        for(int i=0; i<this->ranks.size(); i++){
            this->indices.push_back(post_indices[this->ranks[i]]);
        }
        post_indices.clear();

        this->w = std::vector< std::vector< std::vector< double > > >();
        this->record_w = false;


    };

    std::vector <int> indices;

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

        if(this->record_w && ( (t - this->offset_) % this->period_ == this->period_offset_ )){
            std::vector< std::vector< double > > tmp;
            for(int i=0; i<this->ranks.size(); i++){
                tmp.push_back(std::move(proj0.get_matrix_variable_row<double>(proj0.w, this->indices[i])));
            }
            this->w.push_back(tmp);
            tmp.clear();
        }

    };

    void record_targets() { /* nothing to do here */ }
    long int size_in_bytes() {
        std::cout << "ProjMonitor::size_in_bytes(): not implemented for openMP paradigm." << std::endl;
        return 0;
    }

    void clear() {
        std::cout << "PopMonitor0::clear(): not implemented for openMP paradigm." << std::endl;
    }


    // Local variable w
    std::vector< std::vector< std::vector< double > > > w ;
    bool record_w ;

};

