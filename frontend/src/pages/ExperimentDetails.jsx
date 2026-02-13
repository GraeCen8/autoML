import { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import axios from 'axios';
import { Loader2, CheckCircle, XCircle, Clock } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const ExperimentDetails = () => {
    const { id } = useParams();
    const [experiment, setExperiment] = useState(null);
    const [results, setResults] = useState([]);
    const [loading, setLoading] = useState(true);

    const fetchData = async () => {
        try {
            const expRes = await axios.get(`http://localhost:8000/experiment/${id}`);
            setExperiment(expRes.data);

            if (expRes.data.status === 'completed') {
                const resRes = await axios.get(`http://localhost:8000/experiment/${id}/results`);
                setResults(resRes.data);
            }
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchData();
        const interval = setInterval(() => {
            if (experiment && (experiment.status === 'running' || experiment.status === 'queued')) {
                fetchData();
            }
        }, 5000);
        return () => clearInterval(interval);
    }, [id, experiment?.status]);


    if (loading) return <div className="flex justify-center p-8"><Loader2 className="animate-spin" /></div>;
    if (!experiment) return <div>Not found</div>;

    return (
        <div className="space-y-8">
            <div className="flex justify-between items-start">
                <div>
                    <h2 className="text-3xl font-bold mb-2">{experiment.name}</h2>
                    <div className="flex items-center space-x-2 text-muted-foreground">
                        <span>Target: {experiment.target}</span>
                        <span>•</span>
                        <span className="capitalize">{experiment.status}</span>
                    </div>
                </div>
                <div>
                    {experiment.status === 'running' && <Loader2 className="animate-spin text-blue-500" size={32} />}
                    {experiment.status === 'completed' && <CheckCircle className="text-green-500" size={32} />}
                    {experiment.status === 'failed' && <XCircle className="text-red-500" size={32} />}
                    {experiment.status === 'pending' && <Clock className="text-gray-500" size={32} />}
                </div>
            </div>

            {experiment.status === 'completed' && results.length > 0 && (
                <div className="grid gap-8 lg:grid-cols-2">
                    <div className="bg-card p-6 rounded-lg border border-border">
                        <h3 className="text-xl font-bold mb-6">Performance Metrics</h3>
                        <div className="space-y-4">
                            <MetricRow label="Accuracy" value={results[0].accuracy} />
                            <MetricRow label="Precision" value={results[0].precision} />
                            <MetricRow label="Recall" value={results[0].recall} />
                            <MetricRow label="F1 Score" value={results[0].f1} />
                        </div>
                    </div>

                    <div className="bg-card p-6 rounded-lg border border-border">
                        <h3 className="text-xl font-bold mb-6">Model Comparison</h3>
                        <div className="h-64 cursor-default">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={results}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                                    <XAxis dataKey="id" stroke="#888" />
                                    <YAxis stroke="#888" />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333' }}
                                        itemStyle={{ color: '#fff' }}
                                    />
                                    <Legend />
                                    <Bar dataKey="accuracy" fill="#3b82f6" name="Accuracy" />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </div>
            )}

            {experiment.status === 'failed' && (
                <div className="p-4 bg-red-900/20 border border-red-900 text-red-400 rounded-md">
                    Experiment execution failed. Please check backend logs.
                </div>
            )}
        </div>
    );
};

const MetricRow = ({ label, value }) => (
    <div className="flex justify-between items-center py-2 border-b border-border last:border-0">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono font-bold text-lg">{(value * 100).toFixed(2)}%</span>
    </div>
);

export default ExperimentDetails;
