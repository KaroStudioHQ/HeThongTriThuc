using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.Mathematics;
using UnityEngine;

public class Main : MonoBehaviour
{

    public static object tanh(float x)
    {
        return math.tanh(x);
    }

    public static float tanh_derivate(float x)
    {
        return 1.0f - math.tanh(x) * math.tanh(x);
    }

    public static float sigmoid(float x)
    {
        return 1 / (1 + Mathf.Exp(-x));
    }
    public static float sigmoid_derivate(float x)
    {
        return sigmoid(x) * (1 - sigmoid(x));
    }

    public static object calculate_fit(object loss)
    {
        var total = loss.Sum();
        var fitnesses = new List<object>();
        foreach (var i in Enumerable.Range(0, loss.Count))
        {
            fitnesses.append(loss[i] / total);
        }
        return fitnesses;
    }

    public static object pair_pop(object iris_data, object pop)
    {
        var weights = new List<object>();
        var loss = new List<object>();
        // for each individual
        foreach (var individual_obj in pop)
        {
            weights.append(new List<object> {
                    individual_obj.weights_input,
                    individual_obj.weights_output
                });
            // append 1/sum(MSEs) of individual to list of pop errors
            loss.append(individual_obj.sum_loss(data: iris_data));
        }
        // fitnesses are a fraction of the total error
        var fitnesses = calculate_fit(loss);
        foreach (var i in Enumerable.Range(0, Convert.ToInt32(pop_size * 0.15)))
        {
            Console.WriteLine(i.ToString().zfill(2), "1/sum(MSEs)", loss[i].ToString().rjust(15), (Convert.ToInt32(loss[i] * graphical_error_scale) * "-").ToString().rjust(20), "fitness".rjust(12), fitnesses[i].ToString().rjust(17), (Convert.ToInt32(fitnesses[i] * 1000) * "-").ToString().rjust(20));
        }
        WONKO_del(pop);
        // Weight becomes item [0] and fitness [1] in this way, fitness is paired with its weight in a tuple
        return zip(weights, loss, fitnesses);
    }

    public static object roulette(object fitness_scores)
    {
        var cumalative_fitness = 0.0;
        var r = random.random();
        // Fitness score for each chromosome
        foreach (var i in Enumerable.Range(0, fitness_scores.Count))
        {
            // Fitness scores are added for each chromosome to accrue fitness
            cumalative_fitness += fitness_scores[i];
            // The colorimetric index is returned if the cumulative fitness is greater than r
            if (cumalative_fitness > r)
            {
                return i;
            }
        }
    }

    public static object iterate_pop(object ranked_pop)
    {
        var ranked_weights = (from item in ranked_pop
                              select item[0]).ToList();
        var fitness_scores = (from item in ranked_pop
                              select item[-1]).ToList();
        var new_pop_weight = (from x in ranked_weights[::int((pop_size * 0.15))]
                              select eval(repr(x))).ToList();
        // Reproduce two randomly selected, but different chromos, until pop_size is reached
        while (new_pop_weight.Count <= pop_size)
        {
            var ch1 = new List<object>();
            var ch2 = new List<object>();
            var index1 = roulette(fitness_scores);
            var index2 = roulette(fitness_scores);
            while (index1 == index2)
            {
                // Make sure different chromosomes are used for breeding
                index2 = roulette(fitness_scores);
            }
            // index1, index2 = 3,4
            ch1.extend(eval(repr(ranked_weights[index1])));
            ch2.extend(eval(repr(ranked_weights[index2])));
            if (random.random() < crossover_rate)
            {
                var _tup_1 = crossover(ch1, ch2);
                ch1 = _tup_1.Item1;
                ch2 = _tup_1.Item2;
            }
            mutate(ch1);
            mutate(ch2);
            new_pop_weight.append(ch1);
            new_pop_weight.append(ch2);
        }
        return new_pop_weight;
    }

    public static object crossover(object m1, object m2)
    {
        // ni*nh+nh*no  = total weights
        var r = random.randint(0, nodes_input * nodes_hidden + nodes_hidden * nodes_output);
        var output1 = new List<object> {
                new List<object> {
                    new List<object> {
                        0.0
                    } * nodes_hidden
                } * nodes_input,
                new List<object> {
                    new List<object> {
                        0.0
                    } * nodes_output
                } * nodes_hidden
            };
        var output2 = new List<object> {
                new List<object> {
                    new List<object> {
                        0.0
                    } * nodes_hidden
                } * nodes_input,
                new List<object> {
                    new List<object> {
                        0.0
                    } * nodes_output
                } * nodes_hidden
            };
        foreach (var i in Enumerable.Range(0, m1.Count))
        {
            foreach (var j in Enumerable.Range(0, m1[i].Count))
            {
                foreach (var k in Enumerable.Range(0, m1[i][j].Count))
                {
                    if (r >= 0)
                    {
                        output1[i][j][k] = m1[i][j][k];
                        output2[i][j][k] = m2[i][j][k];
                    }
                    else if (r < 0)
                    {
                        output1[i][j][k] = m2[i][j][k];
                        output2[i][j][k] = m1[i][j][k];
                    }
                    r -= 1;
                }
            }
        }
        return Tuple.Create(output1, output2);
    }

    public static object mutate(object m)
    {
        // A constant can be included to control how much the weight has been abruptly changed
        foreach (var i in Enumerable.Range(0, m.Count))
        {
            foreach (var j in Enumerable.Range(0, m[i].Count))
            {
                foreach (var k in Enumerable.Range(0, m[i][j].Count))
                {
                    if (random.random() < mutation_rate)
                    {
                        m[i][j][k] = random.uniform(-2.0, 2.0);
                    }
                }
            }
        }
    }

    public static object rank_pop(object new_pop_weight, object pop)
    {
        // The new neural network is assigned to the pop_size list
        var loss = new List<object>();
        var copy = new List<object>();
        pop = (from _ in Enumerable.Range(0, pop_size)
               select NeuralNetwork(nodes_input, nodes_hidden, nodes_output)).ToList();
        foreach (var i in Enumerable.Range(0, pop_size))
        {
            copy.append(new_pop_weight[i]);
        }
        foreach (var i in Enumerable.Range(0, pop_size))
        {
            // Everyone is assigned the weight generated by the previous iteration
            pop[i].assign_weights(new_pop_weight, i);
            pop[i].test_weights(new_pop_weight, i);
        }
        foreach (var i in Enumerable.Range(0, pop_size))
        {
            pop[i].test_weights(new_pop_weight, i);
        }
        // Calculate the fitness of these weights and modify them with weights
        var paired_pop = pair_pop(iris_train_data, pop);
        // The weights are sorted in descending order of fitness (most suitable)
        var ranked_pop = paired_pop.OrderByDescending(itemgetter(-1)).ToList();
        loss = (from x in ranked_pop
                select eval(repr(x[1]))).ToList();
        return Tuple.Create(ranked_pop, eval(repr(ranked_pop[0][1])), float(loss.Sum()) / float(loss.Count));
    }

    public static void randomize_matrix(float[][] matrix, float a, float b)
    {
        for (int i = 0; i < matrix.Length; i++)
        {
            for (int j = 0; j < matrix[0].Length; j++)
            {
                matrix[i][j] = UnityEngine.Random.Range(a, b);
            }
        }
    }


    public class NeuralNetwork
    {
        private int nodes_input;
        private int nodes_hidden;
        private int nodes_output;
        private object weights_input;
        private object weights_output;

        public NeuralNetwork(int nodes_input, int nodes_hidden, int nodes_output, string activation_fun = "tanh")
        {
            // number of input, hidden, and output nodes
            this.nodes_input = nodes_input;
            this.nodes_hidden = nodes_hidden;
            this.nodes_output = nodes_output;
            // activations for nodes
            
            // create weights
            
            randomize_matrix(this.weights_input, -0.1, 0.1);
            randomize_matrix(this.weights_output, -2.0, 2.0);
            // define activation function
            if (object.ReferenceEquals(activation_fun, "tanh"))
            {
                //this.activation_fun = tanh;
                //this.activation_fun_deriv = tanh_derivate;
            }
            else if (object.ReferenceEquals(activation_fun, "sigmoid"))
            {
                //this.activation_fun = sigmoid;
                //this.activation_fun_deriv = sigmoid_derivate;
            }
        }

        public virtual object sum_loss(object data)
        {
            var loss = 0.0;
            foreach (var item in data)
            {
                var inputs = item[0];
                var targets = item[1];
                this.feed_forward(inputs);
                loss += this.calculate_loss(targets);
            }
            var inverr = 1.0 / loss;
            return inverr;
        }

        public virtual object calculate_loss(object targets)
        {
            var loss = 0.0;
            foreach (var k in Enumerable.Range(0, targets.Count))
            {
                loss += 0.5 * Math.Pow(targets[k] - this.activations_output[k], 2);
            }
            return loss;
        }

        public virtual object feed_forward(List<float> inputs)
        {
            if (inputs.Count != this.nodes_input)
            {
                Console.WriteLine("incorrect number of inputs");
            }
            foreach (var i in Enumerable.Range(0, this.nodes_input))
            {
                this.activations_input[i] = inputs[i];
            }
            foreach (var j in Enumerable.Range(0, this.nodes_hidden))
            {
                this.activations_hidden[j] = this.activation_fun((from i in Enumerable.Range(0, this.nodes_input)
                                                                  select (this.activations_input[i] * this.weights_input[i][j])).ToList().Sum());
            }
            foreach (var k in Enumerable.Range(0, this.nodes_output))
            {
                this.activations_output[k] = this.activation_fun((from j in Enumerable.Range(0, this.nodes_hidden)
                                                                  select (this.activations_hidden[j] * this.weights_output[j][k])).ToList().Sum());
            }
            return this.activations_output;
        }

        public virtual object assign_weights(object weights, object I)
        {
            var io = 0;
            foreach (var i in Enumerable.Range(0, this.nodes_input))
            {
                foreach (var j in Enumerable.Range(0, this.nodes_hidden))
                {
                    this.weights_input[i][j] = weights[I][io][i][j];
                }
            }
            io = 1;
            foreach (var j in Enumerable.Range(0, this.nodes_hidden))
            {
                foreach (var k in Enumerable.Range(0, this.nodes_output))
                {
                    this.weights_output[j][k] = weights[I][io][j][k];
                }
            }
        }

        public virtual object test_weights(object weights, object I)
        {
            var same = new List<object>();
            var io = 0;
            foreach (var i in Enumerable.Range(0, this.nodes_input))
            {
                foreach (var j in Enumerable.Range(0, this.nodes_hidden))
                {
                    if (this.weights_input[i][j] != weights[I][io][i][j])
                    {
                        same.append(("I", i, j, round(this.weights_input[i][j], 2), round(weights[I][io][i][j], 2), round(this.weights_input[i][j] - weights[I][io][i][j], 2)));
                    }
                }
            }
            io = 1;
            foreach (var j in Enumerable.Range(0, this.nodes_hidden))
            {
                foreach (var k in Enumerable.Range(0, this.nodes_output))
                {
                    if (this.weights_output[j][k] != weights[I][io][j][k])
                    {
                        same.append((("O", j, k), round(this.weights_output[j][k], 2), round(weights[I][io][j][k], 2), round(this.weights_output[j][k] - weights[I][io][j][k], 2)));
                    }
                }
            }
            if (same)
            {
                Console.WriteLine(same);
            }
        }

        public virtual object test(object data)
        {
            object result;
            var results = new List<object>();
            var targets = new List<object>();
            foreach (var d in data)
            {
                var inputs = d[0];
                var rounded = (from i in this.feed_forward(inputs)
                               select round(i)).ToList();
                if (rounded == d[1])
                {
                    result = "√ Classification Prediction is Correct ";
                }
                else
                {
                    result = "× Classification Prediction is Wrong";
                }
                Console.WriteLine("{0} {1} {2} {3} {4} {5} {6}".format("Inputs:", d[0], "-->", this.feed_forward(inputs).ToString().rjust(65), "target classification", d[1], result));
                results += this.feed_forward(inputs);
                targets += d[1];
            }
            return Tuple.Create(results, targets);
        }
    }
}
