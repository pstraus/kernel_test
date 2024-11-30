#include <iostream>

namespace logger
{
    enum class LOG_LEVEL
    {
        trace,
        debug,
        warning,
        error,
        fatal   
    };

    class HostLogger
    {
        public:
            HostLogger(std::string_view logname);
            ~HostLogger();
            void LOG();
        private:
         
    };

    HostLogger::HostLogger(std::string_view logname)
    {
        //open file

    }
    HostLogger::~HostLogger()
    {
        // Close file handle
    }
    void HostLogger::LOG()
    {
        // Gotta establish a stream or something
    }
}